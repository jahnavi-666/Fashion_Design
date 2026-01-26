import numpy as np
import faiss
import os

def build_faiss_index(
    embeddings_path="models/embeddings.npy",
    index_path="models/faiss.index"
):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("Embeddings file not found. Run embeddings.py first.")

    # Load embeddings
    x = np.load(embeddings_path).astype("float32")

    dim = x.shape[1]

    # Create FAISS Index
    index = faiss.IndexFlatL2(dim)

    # Add embeddings
    index.add(x)

    # Save index
    faiss.write_index(index, index_path)

    print(f"FAISS index built and saved to {index_path}")


# Allow running as script
if __name__ == "__main__":
    build_faiss_index()
