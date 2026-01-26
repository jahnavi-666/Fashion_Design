import React, { useRef, useEffect, useState } from "react";

export default function TryOn() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [net, setNet] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let mounted = true;
    const loadNet = async () => {
      try {
        const loaded = await window.bodyPix.load();
        if (!mounted) return;
        setNet(loaded);
      } catch (err) {
        console.error("BodyPix load failed", err);
      }
    };
    loadNet();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    let rafId;
    const start = async () => {
      if (!net) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = stream;
      await video.play();

      setRunning(true);

      const render = async () => {
        const seg = await net.segmentPerson(video, {
          internalResolution: "medium",
          segmentationThreshold: 0.7
        });

        // draw live video
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // create mask - we will tint the torso area lightly and draw a test garment rectangle
        const coloredPartImage = window.bodyPix.toMask(seg);
        // put mask on top with globalAlpha
        ctx.putImageData(coloredPartImage, 0, 0);
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = "rgba(0,0,0,0.2)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;

        // For demo: draw a semi-transparent rectangle (fake garment) around center
        const tx = canvas.width * 0.25;
        const ty = canvas.height * 0.25;
        const tw = canvas.width * 0.5;
        const th = canvas.height * 0.45;
        ctx.fillStyle = "rgba(200,50,50,0.5)";
        ctx.fillRect(tx, ty, tw, th);

        rafId = requestAnimationFrame(render);
      };

      render();
    };

    if (net && !running) start();

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [net, running]);

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4">Try-on (Client-side)</h2>
      <p className="text-slate-600 mb-4">This uses TensorFlow.js BodyPix to segment the person. For realistic garment warping you'd need a server-side model (VITON / CP-VTON).</p>

      <div className="flex gap-4">
        <video ref={videoRef} width="320" height="240" style={{ display: "none" }} />
        <canvas ref={canvasRef} width="640" height="480" className="card" />
      </div>
    </div>
  );
}
