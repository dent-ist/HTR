import { type Ref, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { Line, Point } from "../api/types";
import {
  type Size,
  type Viewport,
  ensureVisible,
  fitWidth,
  hitTestLine,
  panBy,
  polygonBox,
  screenToWorld,
  wheelFactor,
  worldToScreen,
  zoomAt,
} from "./viewport";

export interface PageCanvasHandle {
  fitWidth(): void;
}

interface Props {
  imageUrl: string;
  imageSize: Size;
  lines: Line[];
  selectedId: number | null;
  onSelect(id: number): void;
  ref?: Ref<PageCanvasHandle>;
}

const COLORS = {
  plain: "#2f6fdb",
  groundTruth: "#1c9a4a",
  ignored: "#8a8a8a",
  selected: "#f08a00",
};

const DRAG_THRESHOLD = 4;

function tracePolygon(ctx: CanvasRenderingContext2D, vp: Viewport, poly: Point[]): void {
  ctx.beginPath();
  poly.forEach((p, i) => {
    const [x, y] = worldToScreen(vp, p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
}

export function PageCanvas({ imageUrl, imageSize, lines, selectedId, onSelect, ref }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [view, setView] = useState<Size>({ width: 0, height: 0 });
  const [vp, setVp] = useState<Viewport>({ scale: 1, tx: 0, ty: 0 });
  const fittedFor = useRef<string | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean; id: number } | null>(null);

  useImperativeHandle(ref, () => ({ fitWidth: () => setVp(fitWidth(imageSize, view)) }), [imageSize, view]);

  // Load the working copy.
  useEffect(() => {
    let live = true;
    setImage(null);
    setLoadError(false);
    const img = new Image();
    img.onload = () => live && setImage(img);
    img.onerror = () => live && setLoadError(true);
    img.src = imageUrl;
    return () => {
      live = false;
    };
  }, [imageUrl]);

  // Track the element size.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setView({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fit to width once per page, then keep the selected line visible.
  const selected = lines.find((l) => l.id === selectedId) ?? null;
  useEffect(() => {
    if (!image || view.width === 0) return;
    const key = imageUrl;
    if (fittedFor.current !== key) {
      fittedFor.current = key;
      const base = fitWidth(imageSize, view);
      setVp(selected ? ensureVisible(base, polygonBox(selected.polygon), view) : base);
      return;
    }
    if (selected) setVp((cur) => ensureVisible(cur, polygonBox(selected.polygon), view));
    // Only re-run on selection / geometry changes, not on every pan.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image, imageUrl, selectedId, view.width, view.height]);

  // Wheel zoom at the cursor needs a non-passive listener to prevent page scroll.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const factor = wheelFactor(e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY);
      setVp((cur) => zoomAt(cur, e.clientX - r.left, e.clientY - r.top, factor));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Draw.
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || view.width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(view.width * dpr);
    canvas.height = Math.round(view.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#d9dde3";
    ctx.fillRect(0, 0, view.width, view.height);
    if (image) {
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(image, vp.tx, vp.ty, imageSize.width * vp.scale, imageSize.height * vp.scale);
    }
    ctx.lineJoin = "round";
    for (const l of lines) {
      if (l.id === selectedId) continue;
      const color = l.ignored ? COLORS.ignored : l.is_ground_truth ? COLORS.groundTruth : COLORS.plain;
      tracePolygon(ctx, vp, l.polygon);
      ctx.fillStyle = color + (l.ignored ? "14" : "1f");
      ctx.fill();
      ctx.setLineDash(l.ignored ? [6, 4] : []);
      ctx.lineWidth = l.is_ground_truth ? 2 : 1.25;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const sel = lines.find((l) => l.id === selectedId);
    if (sel) {
      tracePolygon(ctx, vp, sel.polygon);
      ctx.fillStyle = COLORS.selected + "40";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = COLORS.selected;
      ctx.setLineDash(sel.ignored ? [6, 4] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [view, vp, image, imageSize, lines, selectedId]);

  useEffect(() => {
    draw();
  }, [draw]);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    d.moved = true;
    d.x = e.clientX;
    d.y = e.clientY;
    setVp((cur) => panBy(cur, dx, dy));
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.moved) return;
    const r = e.currentTarget.getBoundingClientRect();
    const world = screenToWorld(vp, [e.clientX - r.left, e.clientY - r.top]);
    const hit = hitTestLine(lines, world);
    if (hit !== null) onSelect(hit);
  };

  return (
    <div className="canvas-wrap" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        className="page-canvas"
        style={{ width: view.width, height: view.height }}
        role="img"
        aria-label={`Page image with ${lines.length} text lines. Wheel to zoom, drag to pan, click a line to select it.`}
        data-testid="page-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (drag.current = null)}
      />
      {!image && !loadError && <div className="canvas-note">Loading page image...</div>}
      {loadError && <div className="canvas-note error">Could not load the page image.</div>}
    </div>
  );
}
