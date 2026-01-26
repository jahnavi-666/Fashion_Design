import React from "react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto">
      <header className="mb-8">
        <h1 className="text-4xl font-extrabold mb-2">Welcome to AI Fashion Stylist</h1>
        <p className="text-slate-600">Upload a photo or select items from your closet to get outfit recommendations.</p>
      </header>

      <div className="grid grid-cols-3 gap-6">
        <Link to="/closet" className="card p-6 text-center">
          <div className="text-2xl font-semibold mb-2">Closet</div>
          <div className="text-slate-500">Add and manage your wardrobe items.</div>
        </Link>

        <Link to="/outfits" className="card p-6 text-center">
          <div className="text-2xl font-semibold mb-2">Outfits</div>
          <div className="text-slate-500">Get outfit suggestions from a single photo.</div>
        </Link>

        <Link to="/tryon" className="card p-6 text-center">
          <div className="text-2xl font-semibold mb-2">Try-on</div>
          <div className="text-slate-500">Try garments in augmented reality (browser).</div>
        </Link>
      </div>
    </div>
  );
}
