#uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
import os
import io
import csv
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import faiss
import uvicorn

from app.model_utils import load_backbone, image_to_embedding


# ============================================
# PATH SETUP
# ============================================

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_ROOT, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USER_CLOSET_DIR = os.path.join(DATA_DIR, "user_closet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

USER_EMB_PATH = os.path.join(MODELS_DIR, "user_embeddings.npy")
USER_META_PATH = os.path.join(MODELS_DIR, "user_meta.csv")
USER_FAISS_PATH = os.path.join(MODELS_DIR, "user_faiss.index")

os.makedirs(USER_CLOSET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Add these imports near the top of main.py
import torch
import numpy as np
from app.clip_utils import image_to_clip_embedding
from app.compat_model import CompatNet

# Device and compat model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPAT_MODEL_PATH = os.path.join(MODELS_DIR, "compat_model.pth")

# Load trained compatibility model (safe for CPU if no GPU)
compat_model = None
if os.path.exists(COMPAT_MODEL_PATH):
    compat_model = CompatNet(emb_dim=512)  # CLIP dim
    state = torch.load(COMPAT_MODEL_PATH, map_location=DEVICE)
    compat_model.load_state_dict(state)
    compat_model.to(DEVICE)
    compat_model.eval()
    print("Loaded compat_model from", COMPAT_MODEL_PATH)
else:
    print("compat_model.pth not found in models/. Upload your trained file.")

# ============================================
# FASTAPI + CORS
# ============================================

app = FastAPI(title="Fashion Stylist - User Closet System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve user closet images
app.mount("/closet", StaticFiles(directory=USER_CLOSET_DIR), name="closet")


# ============================================
# LOAD EMBEDDING MODEL
# ============================================

EMBED_MODEL = load_backbone()

def get_embed_dim():
    dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
    emb = image_to_embedding(EMBED_MODEL, dummy)
    return emb.shape[0]

EMBED_DIM = get_embed_dim()


# ============================================
# In-memory FAISS index + metadata
# ============================================

user_index = None
user_meta = []
index_lock = threading.Lock()


def load_user_index():
    """Load stored embeddings + metadata if present."""
    global user_index, user_meta

    # Load metadata
    if os.path.exists(USER_META_PATH):
        with open(USER_META_PATH, newline='', encoding='utf-8') as f:
            user_meta = list(csv.DictReader(f))
    else:
        user_meta = []

    # Load embedding matrix
    if os.path.exists(USER_EMB_PATH):
        emb = np.load(USER_EMB_PATH).astype("float32")
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        user_index = idx

        # Save FAISS to disk
        faiss.write_index(idx, USER_FAISS_PATH)
        print("Loaded index with", idx.ntotal, "items")

    else:
        # Empty FAISS index
        idx = faiss.IndexFlatL2(EMBED_DIM)
        user_index = idx
        print("Created empty FAISS index")


try:
    load_user_index()
except Exception as e:
    print("Error loading user index:", e)
    user_index = faiss.IndexFlatL2(EMBED_DIM)
    user_meta = []


# ============================================
# Helper to save uploaded item
# ============================================

def append_user_item(filename: str, category: str, color: str = "", tags: str = ""):
    """
    Save metadata (append) and embedding (append) and update in-memory FAISS index.
    Also compute & persist CLIP embedding per user item for compatibility model.
    """
    global user_index, user_meta

    filepath = os.path.join(USER_CLOSET_DIR, filename)
    img = Image.open(filepath).convert("RGB")

    # --- existing embedding (your backbone) ---
    emb = image_to_embedding(EMBED_MODEL, img).astype('float32')  # shape (d,)
    emb_row = emb.reshape(1, -1)

    with index_lock:
        # update persistent embeddings file (backbone)
        if os.path.exists(USER_EMB_PATH):
            existing = np.load(USER_EMB_PATH)
            new_emb_matrix = np.vstack([existing, emb_row])
        else:
            new_emb_matrix = emb_row
        np.save(USER_EMB_PATH, new_emb_matrix)

        # compute CLIP embedding and persist aligned array
        clip_emb = image_to_clip_embedding(img)  # numpy float32 (512,)
        USER_CLIP_EMB_PATH = os.path.join(MODELS_DIR, "user_clip_embeddings.npy")
        if os.path.exists(USER_CLIP_EMB_PATH):
            existing_clip = np.load(USER_CLIP_EMB_PATH)
            new_clip_matrix = np.vstack([existing_clip, clip_emb.reshape(1, -1)])
        else:
            new_clip_matrix = clip_emb.reshape(1, -1)
        np.save(USER_CLIP_EMB_PATH, new_clip_matrix)

        # Append to metadata file
        meta_row = {
            "filename": filename,
            "category": category,
            "color": color,
            "tags": tags,
            "source_url": f"http://localhost:8000/closet/{filename}"
        }
        write_header = not os.path.exists(USER_META_PATH)
        with open(USER_META_PATH, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["filename","category","color","tags","source_url"])
            if write_header:
                writer.writeheader()
            writer.writerow(meta_row)

        # update in-memory meta and faiss index
        user_meta.append(meta_row)
        try:
            if user_index is None or user_index.ntotal == 0:
                # build fresh index from new_emb_matrix
                d = new_emb_matrix.shape[1]
                idx = faiss.IndexFlatL2(d)
                idx.add(new_emb_matrix.astype('float32'))
                user_index = idx
            else:
                user_index.add(emb_row.astype('float32'))
        except Exception as e:
            # fallback: rebuild index fully
            emb_matrix = np.load(USER_EMB_PATH).astype('float32')
            idx = faiss.IndexFlatL2(emb_matrix.shape[1])
            idx.add(emb_matrix)
            user_index = idx

        # persist index
        faiss.write_index(user_index, USER_FAISS_PATH)

    return meta_row


# ============================================
# API ENDPOINTS
# ============================================

from fastapi import FastAPI, UploadFile, File, Form

@app.post("/add_to_closet/")
async def add_to_closet(
    file: UploadFile = File(...),
    category: str = Form("unknown"),
    color: str = Form(""),
    tags: str = Form("")
):
    safe_name = file.filename.replace(" ", "_")
    out_path = os.path.join(USER_CLOSET_DIR, safe_name)

    # save image
    contents = await file.read()
    with open(out_path, "wb") as f:
        f.write(contents)

    # save metadata + embedding
    meta = append_user_item(safe_name, category, color, tags)

    return {
        "status": "saved",
        "url": meta["source_url"],
        "meta": meta
    }

@app.get("/list_closet/")
def list_closet():
    """Return all closet items."""
    if os.path.exists(USER_META_PATH):
        with open(USER_META_PATH, newline="", encoding="utf-8") as f:
            return {"items": list(csv.DictReader(f))}
    return {"items": []}


from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

@app.post("/upload_query/")
async def upload_query(file: UploadFile = File(...), top_k: int = 100):
    """
    Improved outfit recommendation:
    - compute backbone embedding (FAISS shortlist)
    - compute CLIP query embedding
    - gather ALL candidate indices in the needed categories (guaranteed)
    - vectorized scoring with compat_model
    - small embed_sim + color boosters
    - return top-3 per target category
    """
    global user_index, user_meta

    if user_index is None or user_index.ntotal == 0:
        return JSONResponse({"message": "Closet empty. Upload items first."}, status_code=400)

    contents = await file.read()
    q_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # backbone embedding used for FAISS shortlist (optional)
    q_back = image_to_embedding(EMBED_MODEL, q_img).astype("float32").reshape(1, -1)

    # CLIP embedding for compat scoring
    q_clip = image_to_clip_embedding(q_img)  # numpy (512,)

    # FAISS shortlist (may be useful), but we won't trust it exclusively
    with index_lock:
        k = min(max(20, top_k), max(20, user_index.ntotal))
        D, I = user_index.search(q_back, k)

    # Get detected category (nearest neighbor)
    nearest_idx = int(I[0][0])
    uploaded_category = user_meta[nearest_idx].get("category", "unknown")

    # Map uploaded -> needed categories
    if uploaded_category == "top":
        needed = ["bottom", "footwear", "accessory"]
    elif uploaded_category == "bottom":
        needed = ["top", "footwear", "accessory"]
    elif uploaded_category == "footwear":
        needed = ["top", "bottom", "accessory"]
    else:
        needed = ["top", "bottom", "footwear"]

    # Load user CLIP embeddings (must exist; created at upload time)
    USER_CLIP_EMB_PATH = os.path.join(MODELS_DIR, "user_clip_embeddings.npy")
    if not os.path.exists(USER_CLIP_EMB_PATH):
        return JSONResponse({"message": "No user CLIP embeddings found. Re-upload closet items."}, status_code=500)
    user_clip_embs = np.load(USER_CLIP_EMB_PATH).astype("float32")  # shape (N_user, 512)

    # 1) Build candidate_indices: prefer FAISS shortlist candidates that match needed categories,
    # but ALWAYS include all items in needed categories (full scan) to guarantee candidates.
    candidate_indices = []
    # a) include FAISS shortlist matches if they are in needed categories
    for idx_np in I[0]:
        idx = int(idx_np)
        if idx < len(user_meta) and user_meta[idx].get("category") in needed:
            candidate_indices.append(idx)

    # b) ensure we include ALL items in needed categories (deduplicate)
    full_category_indices = [i for i, m in enumerate(user_meta) if m.get("category") in needed]
    # Combine while preserving order, dedupe
    combined = list(dict.fromkeys(candidate_indices + full_category_indices))
    candidate_indices = combined

    if len(candidate_indices) == 0:
        # no items in needed categories at all
        return {"detected": uploaded_category, "recommendations": {c: [] for c in needed}}

    # 2) Prepare arrays for vectorized scoring
    cand_clip = user_clip_embs[candidate_indices]  # (M, 512)
    q_clip_rep = np.repeat(q_clip.reshape(1, -1), cand_clip.shape[0], axis=0).astype("float32")

    # 3) Vectorized compat_model scoring in batches
    batch_size = 256
    scores_parts = []
    with torch.no_grad():
        compat_model.to(DEVICE)
        for i in range(0, cand_clip.shape[0], batch_size):
            a_batch = torch.from_numpy(q_clip_rep[i:i+batch_size]).to(DEVICE)
            b_batch = torch.from_numpy(cand_clip[i:i+batch_size]).to(DEVICE)
            s = compat_model(a_batch, b_batch)
            scores_parts.append(s.cpu().numpy())
    scores = np.concatenate(scores_parts, axis=0)  # (M,)

    # 4) Compute small embed similarity boost using FAISS distances (if available)
    idx_to_dist = {}
    for dist_np, idx_np in zip(D[0], I[0]):
        idx_to_dist[int(idx_np)] = float(dist_np)

    # small color booster function (optional - replace with more advanced if you saved color_rgb)
    def color_boost(q_color, cand_color):
        try:
            if not q_color or not cand_color:
                return 0.5
            return 1.0 if str(q_color).lower() == str(cand_color).lower() else 0.6
        except:
            return 0.5

    # 5) Aggregate results per category with combined score
    results_by_cat = {c: [] for c in needed}
    for score_val, c_idx in zip(scores.tolist(), candidate_indices):
        meta = user_meta[c_idx]
        cat = meta.get("category", "unknown")
        # embed sim from FAISS if present else small fallback
        dist = idx_to_dist.get(c_idx, 1e6)
        embed_sim = 1.0 / (1.0 + float(dist))
        # color boost (if you stored color in meta; qcol left blank for now)
        qcol = ""  # if you compute query color, set here
        ccol = meta.get("color", "")
        colb = color_boost(qcol, ccol)
        final_score = float(0.80 * float(score_val) + 0.12 * embed_sim + 0.08 * colb)

        results_by_cat.setdefault(cat, []).append({
            "filename": meta.get("filename", ""),
            "category": cat,
            "source_url": meta.get("source_url", ""),
            "compat": float(score_val),
            "embed_sim": float(embed_sim),
            "score": final_score
        })

    # 6) Sort & take top results (3 per category)
    for c in list(results_by_cat.keys()):
        results_by_cat[c] = sorted(results_by_cat[c], key=lambda x: -x["score"])[:3]

        # --- compatibility: also return a flat `results` array for old frontend ---
    # Flatten recommendations to a simple array of items with category
    flat_results = []
    for cat, items in results_by_cat.items():
        for it in items:
            flat = {
                "filename": it.get("filename"),
                "category": cat,
                "source_url": it.get("source_url"),
                "distance": it.get("embed_sim", 0.0),   # keeps the old "distance" name
                "score": it.get("score", 0.0)
            }
            flat_results.append(flat)

    # Keep both response shapes
    response = {
        "detected": uploaded_category,
        "recommendations": results_by_cat,
        "results": flat_results  # <-- old frontend expects this
    }

    return response

    


@app.get("/info/")
def info():
    return {
        "status": "running",
        "closet_items": len(user_meta),
        "faiss_items": user_index.ntotal,
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
