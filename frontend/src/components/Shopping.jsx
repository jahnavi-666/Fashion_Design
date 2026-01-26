import React from "react";

export default function Shopping() {
  // lightweight shopping view: images link to source_url in metadata
  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4">Shopping Arena</h2>
      <p className="text-slate-600 mb-6">This page can show curated products from your catalog or external links.</p>

      <div className="grid grid-cols-3 gap-6">
        <div className="card overflow-hidden">
          <img src="https://via.placeholder.com/400x300?text=Shop+Item+1" className="img-thumb" alt="" />
          <div className="p-3">
            <div className="font-medium">Casual Shirt</div>
            <div className="text-sm text-slate-500">Rs. 899 • <a href="#" className="text-sky-600">Buy</a></div>
          </div>
        </div>

        <div className="card overflow-hidden">
          <img src="https://via.placeholder.com/400x300?text=Shop+Item+2" className="img-thumb" alt="" />
          <div className="p-3">
            <div className="font-medium">Denim Jeans</div>
            <div className="text-sm text-slate-500">Rs. 1299 • <a href="#" className="text-sky-600">Buy</a></div>
          </div>
        </div>
      </div>
    </div>
  );
}
