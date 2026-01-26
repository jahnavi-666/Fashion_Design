import React, { useEffect, useState } from "react";
import axios from "axios";

export default function Closet() {
  const [items, setItems] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [category, setCategory] = useState("top");

  const fetchCloset = async () => {
    try {
      const resp = await axios.get("http://localhost:8000/list_closet/");
      setItems(resp.data.items || []);
    } catch (err) {
      console.error("fetchCloset error", err);
    }
  };

  useEffect(() => {
    fetchCloset();
  }, []);

  const onUpload = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setUploading(true);
    const fd = new FormData();
    fd.append("file", f, f.name);
    fd.append("category", category);

    try {
      const resp = await axios.post("http://localhost:8000/add_to_closet/", fd, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      // update list
      fetchCloset();
      alert("Uploaded and added to closet");
    } catch (err) {
      alert("Upload failed: " + (err?.message || "error"));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4">Your Closet</h2>

      <div className="mb-4 flex gap-4 items-center">
        <label className="btn cursor-pointer">
          {uploading ? "Uploading..." : "Upload Item"}
          <input type="file" onChange={onUpload} className="hidden" accept="image/*" />
        </label>

        <label className="text-slate-700">Category:</label>
        <select value={category} onChange={(e) => setCategory(e.target.value)} className="p-2 border rounded">
          <option value="top">Top</option>
          <option value="bottom">Bottom</option>
          <option value="footwear">Footwear</option>
          <option value="accessory">Accessory</option>
        </select>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {items.length === 0 && <div className="text-slate-500">No items yet. Upload some items.</div>}
        {items.map((it, idx) => (
          <div key={idx} className="card overflow-hidden">
            <img src={it.source_url} className="img-thumb" alt={it.filename} />
            <div className="p-3">
              <div className="font-medium">{it.filename}</div>
              <div className="text-sm text-slate-500">{it.category}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
