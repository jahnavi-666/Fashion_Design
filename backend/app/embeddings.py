import os
import csv
import pathlib
import numpy as np
from PIL import Image
from app.model_utils import load_backbone, image_to_embedding

BASE = pathlib.Path(__file__).resolve().parent.parent
def build_embeddings(
    #image_folder="data/images",
    #metadata_csv="data/metadata.csv",   # backend/
    image_folder = str(BASE / "data" / "images"),
    metadata_csv = str(BASE / "data" / "metadata.csv"),
    out_npy="models/embeddings.npy",
    out_meta="models/meta.csv"
):
    # Load model
    model = load_backbone()

    # Prepare storage
    embeddings = []
    metadata_rows = []

    # Read metadata.csv
    with open(metadata_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(image_folder, row['filename'])

            if not os.path.exists(img_path):
                print(f"WARNING: Image not found → {img_path}")
                continue

            # Load image
            img = Image.open(img_path).convert('RGB')

            # Generate embedding
            emb = image_to_embedding(model, img)

            embeddings.append(emb)
            metadata_rows.append(row)

    # Convert to numpy array
    embeddings = np.stack(embeddings)

    # Create models folder if missing
    os.makedirs("models", exist_ok=True)

    # Save embeddings
    np.save(out_npy, embeddings)
    print(f"Embeddings saved to {out_npy}")

    # Save filtered metadata
    keys = metadata_rows[0].keys()
    with open(out_meta, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(keys))
        writer.writeheader()
        for r in metadata_rows:
            writer.writerow(r)

    print(f"Metadata saved to {out_meta}")


# Allow running as script
if __name__ == "__main__":
    build_embeddings()
