import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import Home from "./components/Home";
import Closet from "./components/Closet";
import Outfits from "./components/Outfits";
import Shopping from "./components/Shopping";
import TryOn from "./components/TryOn";

export default function App() {
  return (
    <div>
      <nav className="p-4 flex items-center justify-between bg-white shadow">
        <div className="text-xl font-bold">AI Fashion Stylist</div>
        <div className="flex gap-4">
          <Link to="/" className="text-slate-700">Home</Link>
          <Link to="/closet" className="text-slate-700">Closet</Link>
          <Link to="/outfits" className="text-slate-700">Outfits</Link>
          <Link to="/shopping" className="text-slate-700">Shopping</Link>
          <Link to="/tryon" className="text-slate-700">Try-on</Link>
        </div>
      </nav>

      <main className="p-6">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/closet" element={<Closet />} />
          <Route path="/outfits" element={<Outfits />} />
          <Route path="/shopping" element={<Shopping />} />
          <Route path="/tryon" element={<TryOn />} />
        </Routes>
      </main>
    </div>
  );
}
