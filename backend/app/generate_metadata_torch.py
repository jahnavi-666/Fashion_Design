import os
import time
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import multiprocessing as mp
from sklearn.cluster import MiniBatchKMeans

IMAGE_DIR = "data/images"
OUT_EMB = "data/embeddings_torch.npy"
OUT_FILES = "data/filenames.npy"
OUT_META_CLUSTER = "data/metadata_clusters.csv"

BATCH_SIZE = 64
NUM_WORKERS = max(1, mp.cpu_count() - 1)
INPUT_SIZE = (160, 160)

# ------------------------
# LOAD FAST MODEL
# ------------------------
def load_model():
    base = models.mobilenet_v2(pretrained=True)
    base.classifier = torch.nn.Identity()   # remove classifier, keep features
    base.eval()
    return base

# ------------------------
# PREPROCESS
# ------------------------
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img.numpy()

# ------------------------
# WORKER INITIALIZER
# ------------------------
global_model = None
def worker_init():
    global global_model
    global_model = load_model()

# ------------------------
# PROCESS CHUNK
# ------------------------
def process_chunk(paths):
    global global_model
    model = global_model

    feats = []
    files_out = []

    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch_imgs = [preprocess_image(p) for p in batch_paths]
        batch = torch.tensor(batch_imgs)

        with torch.no_grad():
            out = model(batch)
            out = out.numpy()

        feats.append(out)
        files_out.extend(batch_paths)

    if len(feats) == 0:
        return np.zeros((0,1280),dtype=np.float32), []

    return np.concatenate(feats,axis=0), files_out

# ------------------------
# CHUNKIFY
# ------------------------
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    chunks = []
    i = 0
    for _ in range(n):
        size = k + (1 if m > 0 else 0)
        m -= 1
        chunks.append(lst[i:i+size])
        i += size
    return [c for c in chunks if c]

# ------------------------
# MAIN
# ------------------------
def main():
    t0 = time.time()
    files = [os.path.join(IMAGE_DIR,f) for f in os.listdir(IMAGE_DIR)
             if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    N = len(files)
    print(f"Found {N} images.")

    num_workers = min(NUM_WORKERS, N)
    chunks = chunkify(files, num_workers)
    print(f"Using {num_workers} workers, batch_size={BATCH_SIZE}")

    with mp.Pool(processes=num_workers, initializer=worker_init) as pool:
        results = pool.map(process_chunk, chunks)

    feats_list = [r[0] for r in results if len(r[0])>0]
    files_list = []
    for r in results:
        files_list.extend(r[1])

    embeddings = np.concatenate(feats_list,axis=0).astype(np.float32)
    print("Embeddings:", embeddings.shape)

    np.save(OUT_EMB, embeddings)
    np.save(OUT_FILES, np.array([os.path.basename(p) for p in files_list]))
    print("Saved embeddings + filenames.")

    print("Clustering...")
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=4096)
    kmeans.fit(embeddings)
    labels = kmeans.predict(embeddings)

    with open(OUT_META_CLUSTER,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","cluster"])
        writer.writeheader()
        for fn, lbl in zip(np.load(OUT_FILES), labels):
            fn = fn if isinstance(fn,str) else fn.decode()
            writer.writerow({
                "filename": fn,
                "cluster": int(lbl)
            })

    print("Saved metadata_clusters.csv")
    print(f"Total time: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
