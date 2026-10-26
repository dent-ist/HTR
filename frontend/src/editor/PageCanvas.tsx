import { type Ref, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { BAND_COLOR, BAND_DASH, bandOf, isPrediction } from "./confidence";
import type { Line, Point } from "../api/types";
import { IGNORE_OVERLAP, overlapShare } from "./geometry";
import type { Mode } from "./modes";
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

export type CanvasEvent =
  | { type: "click"; world: Point; hitId: number | null; tol: number }
  | { type: "regionStart" | "regionMove" | "regionEnd"; world: Point };

interface Props {
  imageUrl: string;
  imageSize: Size;
  lines: Line[];
  selectedId: number | null;
  mode: Mode;
  onEvent(e: CanvasEvent): void;
  ref?: Ref<PageCanvasHandle>;
}

const COLORS = {
  plain: "#2f6fdb",
  groundTruth: "#1c9a4a",
  ignored: "#8a8a8a",
  selected: "#f08a00",
  picked: "#8e24aa",
  ghost: "#d81b60",
  cutLeft: "#00897b",
  cutRight: "#5e35b1",
};

const DRAG_THRESHOLD = 4;
const CLICK_TOLERANCE_PX = 8;

function traceLine(ctx: CanvasRenderingContext2D, vp: Viewport, pts: Point[], close: boolean): void {
  ctx.beginPath();
  pts.forEach((p, i) => {
    const [x, y] = worldToScreen(vp, p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (close) ctx.closePath();
}

export function PageCanvas({ imageUrl, imageSize, lines, selectedId, mode, onEvent, ref }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [view, setView] = useState<Size>({ width: 0, height: 0 });
  const [vp, setVp] = useState<Viewport>({ scale: 1, tx: 0, ty: 0 });
  const fittedFor = useRef<string | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean; region: boolean } | null>(null);

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
  }, [image, imageUrl, selectedId, selected?.polygon, view.width, view.height]);

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

  // Expose the viewport to tests/screenshots (read-only data attributes on the canvas).
  const canvasData = { "data-scale": vp.scale.toFixed(4), "data-tx": vp.tx.toFixed(1), "data-ty": vp.ty.toFixed(1) };

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
    const picked = mode.kind === "merge" ? new Set(mode.picked) : new Set<number>();
    const region = mode.kind === "region" ? mode.rect : null;
    const wouldIgnore = region
      ? new Set(lines.filter((l) => !l.ignored && overlapShare(l.polygon, region) >= IGNORE_OVERLAP).map((l) => l.id))
      : new Set<number>();
    const ghosted = (id: number) => picked.has(id) || wouldIgnore.has(id);

    for (const l of lines) {
      if (l.id === selectedId || ghosted(l.id)) continue;
      const band = isPrediction(l) ? bandOf(l.confidence!) : null;
      const color = l.ignored ? COLORS.ignored : l.is_ground_truth ? COLORS.groundTruth : band ? BAND_COLOR[band] : COLORS.plain;
      traceLine(ctx, vp, l.polygon, true);
      ctx.fillStyle = color + (l.ignored ? "14" : "1f");
      ctx.fill();
      ctx.setLineDash(l.ignored ? [6, 4] : band ? BAND_DASH[band] : []);
      ctx.lineWidth = l.is_ground_truth ? 2 : 1.25;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const sel = lines.find((l) => l.id === selectedId);
    if (sel && !ghosted(sel.id)) {
      traceLine(ctx, vp, sel.polygon, true);
      ctx.fillStyle = COLORS.selected + "40";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = COLORS.selected;
      ctx.setLineDash(sel.ignored ? [6, 4] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // ---- ghost geometry for the active repair mode ----
    for (const l of lines) {
      if (!ghosted(l.id)) continue;
      const isPicked = picked.has(l.id);
      traceLine(ctx, vp, l.polygon, true);
      ctx.fillStyle = (isPicked ? COLORS.picked : COLORS.ghost) + "40";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = isPicked ? COLORS.picked : COLORS.ghost;
      ctx.setLineDash(isPicked ? [] : [6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (mode.kind === "split" && mode.preview) {
      const preview = mode.preview;
      const line = lines.find((l) => l.id === preview.lineId);
      if (line) {
        const [cx] = worldToScreen(vp, [preview.x, 0]);
        ctx.save();
        traceLine(ctx, vp, line.polygon, true);
        ctx.clip();
        ctx.fillStyle = COLORS.cutLeft + "66";
        ctx.fillRect(0, 0, cx, view.height);
        ctx.fillStyle = COLORS.cutRight + "66";
        ctx.fillRect(cx, 0, view.width - cx, view.height);
        ctx.restore();
        const box = polygonBox(line.polygon);
        const top = worldToScreen(vp, [0, box.minY])[1];
        const bottom = worldToScreen(vp, [0, box.maxY])[1];
        ctx.strokeStyle = COLORS.ghost;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, top - 10);
        ctx.lineTo(cx, bottom + 10);
        ctx.stroke();
        const [px, py] = worldToScreen(vp, [preview.x, preview.y]);
        ctx.fillStyle = COLORS.ghost;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (mode.kind === "baseline" && mode.points.length > 0) {
      ctx.strokeStyle = COLORS.ghost;
      ctx.lineWidth = 3;
      if (mode.points.length > 1) {
        traceLine(ctx, vp, mode.points, false);
        ctx.stroke();
      }
      ctx.fillStyle = COLORS.ghost;
      for (const p of mode.points) {
        const [x, y] = worldToScreen(vp, p);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (region) {
      const [x, y] = worldToScreen(vp, [region.x, region.y]);
      ctx.fillStyle = COLORS.ghost + "22";
      ctx.fillRect(x, y, region.w * vp.scale, region.h * vp.scale);
      ctx.strokeStyle = COLORS.ghost;
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(x, y, region.w * vp.scale, region.h * vp.scale);
      ctx.setLineDash([]);
    }
  }, [view, vp, image, imageSize, lines, selectedId, mode]);

  useEffect(() => {
    draw();
  }, [draw]);

  const worldAt = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const r = e.currentTarget.getBoundingClientRect();
    return screenToWorld(vp, [e.clientX - r.left, e.clientY - r.top]);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const region = mode.kind === "region";
    drag.current = { x: e.clientX, y: e.clientY, moved: false, region };
    if (region) onEvent({ type: "regionStart", world: worldAt(e) });
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d) return;
    if (d.region) return onEvent({ type: "regionMove", world: worldAt(e) });
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
    if (!d) return;
    if (d.region) return onEvent({ type: "regionEnd", world: worldAt(e) });
    if (d.moved) return;
    const world = worldAt(e);
    onEvent({ type: "click", world, hitId: hitTestLine(lines, world), tol: CLICK_TOLERANCE_PX / vp.scale });
  };

  return (
    <div className={`canvas-wrap mode-${mode.kind}`} ref={wrapRef}>
      <canvas
        ref={canvasRef}
        className="page-canvas"
        style={{ width: view.width, height: view.height }}
        role="img"
        aria-label={`Page image with ${lines.length} text lines. Wheel to zoom, drag to pan, click a line to select it.`}
        data-testid="page-canvas"
        {...canvasData}
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
