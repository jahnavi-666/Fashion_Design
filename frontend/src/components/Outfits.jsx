import React, { useState } from "react";
import axios from "axios";

export default function Outfits() {
  const [results, setResults] = useState([]); // flat results array for old UI compatibility
  const [grouped, setGrouped] = useState({}); // new structured recommendations
  const [detected, setDetected] = useState("");
  const [loading, setLoading] = useState(false);

  const onFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", file, file.name);

    try {
      const resp = await axios.post("http://localhost:8000/upload_query/?top_k=6", fd, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      // New API: { detected, recommendations: { category: [items] } }
      // Old API: { results: [items] }
      const data = resp.data || {};
      setDetected(data.detected || "");

      if (data.recommendations) {
        setGrouped(data.recommendations);
        // create a flat fallback so UI that expects an array still works
        const flat = [];
        Object.keys(data.recommendations).forEach(cat => {
          data.recommendations[cat].forEach(item => {
            flat.push({ ...item, category: cat });
          });
        });
        setResults(flat);
      } else if (data.results) {
        // old shape
        setResults(data.results || []);
        setGrouped({});
      } else {
        setResults([]);
        setGrouped({});
      }

    } catch (err) {
      console.error("Upload/query error:", err);
      alert("Error: " + (err?.response?.data?.detail || err.message));
      setResults([]);
      setGrouped({});
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <h2 className="text-2xl font-semibold mb-4">Outfits & Recommendations</h2>

      <div className="mb-4">
        <input type="file" accept="image/*" onChange={onFile} />
        {loading && <div className="mt-2 text-slate-500">Searching for similar items...</div>}
      </div>

      {detected && <div className="mb-4 text-slate-600">Detected category: <strong>{detected}</strong></div>}

      {/* If grouped recommendations present, show them by category */}
      {Object.keys(grouped).length > 0 ? (
        <div className="space-y-6">
          {Object.entries(grouped).map(([cat, items]) => (
            <div key={cat}>
              <h3 className="text-xl font-medium capitalize mb-2">{cat} </h3>
              <div className="grid grid-cols-3 gap-4">
                {items.length === 0 ? (
                  <div className="text-slate-500 col-span-3">No matching items found in your closet.</div>
                ) : (
                  items.map((it, idx) => (
                    <div key={idx} className="card p-2 border rounded shadow-sm">
                      <img src={it.source_url || `http://localhost:8000/catalog/${it.filename}`} alt={it.filename} className="w-full h-40 object-cover mb-2" />
                      <div className="text-sm font-medium">{it.filename}</div>
                      <div className="text-xs text-slate-500">score: {Number(it.score || it.compat || 0).toFixed(3)}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        // fallback to flat results (old UI)
        <div className="grid grid-cols-3 gap-6 mt-6">
          {results.length === 0 ? (
            <div className="text-slate-500">Upload a photo to get recommendations (results will appear here).</div>
          ) : (
            results.map((r, idx) => (
              <div key={idx} className="card overflow-hidden">
                <img src={r.source_url || `http://localhost:8000/catalog/${r.filename}`} className="img-thumb" alt={r.filename} />
                <div className="p-3">
                  <div className="font-medium">{r.filename}</div>
                  <div className="text-sm text-slate-500">{r.category} • score: {(r.score || r.distance || 0).toFixed(3)}</div>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
