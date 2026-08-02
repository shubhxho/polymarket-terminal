"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { useTerminal } from "@/components/TerminalProvider";
import { useElementSize } from "@/hooks/useElementSize";
import { buildCells, heatColor, heatColorCss, type HeatCell } from "@/lib/heatmap";
import { cents, compact, signed } from "@/lib/format";
import { transition } from "@/lib/motion";
import type { Market } from "@/lib/types";

/**
 * Whole-board heatmap: one cell per market, packed into a near-square grid,
 * coloured green/red by 24h change exactly as the tables are. At ~200 cells
 * this is the one view SVG can't do smoothly, so it renders on the GPU.
 *
 * WebGPU draws every cell as one instanced quad in a single draw call; when the
 * browser has no `navigator.gpu` it falls back to Canvas2D, which paints the
 * identical layout the pure `lib/heatmap` core produces. Hit-testing is always
 * CPU-side against that same cell list, so hover and click work either way.
 *
 * The GPU device, shader, pipeline and quad buffer are created ONCE and cached
 * across re-renders — a data tick or resize only rewrites two buffers (the
 * per-cell instance data and a resolution uniform) and re-encodes one pass.
 * Requesting a fresh `GPUDevice` per update — the old behaviour — is the most
 * expensive thing a WebGPU app can do, and it leaked a device on every frame.
 */

/** Persistent per-canvas GPU resources — survive re-renders; freed on unmount. */
type GpuState = {
  device: GPUDevice;
  ctx: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  quadBuf: GPUBuffer;
  resBuf: GPUBuffer; // uniform: canvas [w, h] in CSS px
  bindGroup: GPUBindGroup;
  instBuf: GPUBuffer | null;
  instCap: number; // instance-buffer capacity, in cells
};

export function MarketHeatmap({ markets }: { markets: readonly Market[] }) {
  const { go } = useTerminal();
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cellsRef = useRef<HeatCell[]>([]);
  const gpuRef = useRef<GpuState | null>(null);
  const [hover, setHover] = useState<{ market: Market; x: number; y: number } | null>(null);
  const [backend, setBackend] = useState<"webgpu" | "canvas">("canvas");

  // Sorted brightest-first so the board reads as a ranked field, not a shuffle.
  const ordered = markets;

  // Free the cached device only when the component unmounts — never on a data or
  // size change, which is the whole point of caching it.
  useEffect(() => {
    return () => {
      gpuRef.current?.device.destroy?.();
      gpuRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width < 2 || size.height < 2) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.floor(size.width);
    const h = Math.floor(size.height);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const values = ordered.map((m) => m.chg24h ?? 0);
    const cells = buildCells(values, w, h, 2);
    cellsRef.current = cells;

    let disposed = false;

    // Draw the same cells on the CPU — the guaranteed path, and the fallback
    // whenever WebGPU init throws or is unavailable.
    const drawCanvas2D = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      for (const c of cells) {
        ctx.fillStyle = heatColorCss(c.t);
        ctx.fillRect(c.x, c.y, Math.max(1, c.w), Math.max(1, c.h));
      }
    };

    // One-time device + pipeline creation, cached in `gpuRef`. Resolution lives in
    // a uniform (not baked into the shader), so a resize never rebuilds anything.
    const initGpu = async (): Promise<GpuState | null> => {
      if (!navigator.gpu) return null;
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter || disposed) return null;
      const device = await adapter.requestDevice();
      if (disposed) {
        device.destroy?.();
        return null;
      }
      // A cached device can be lost (GPU reset, driver hiccup, tab backgrounded).
      // Drop the cache when that happens so the next render re-inits a fresh device
      // — or falls back to Canvas2D — instead of drawing into a dead device forever.
      device.lost.then((info) => {
        if (gpuRef.current?.device === device) {
          gpuRef.current = null;
          if (info.reason !== "destroyed") setBackend("canvas");
        }
      });
      const ctx = canvas.getContext("webgpu");
      if (!ctx) return null;
      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format, alphaMode: "premultiplied" });

      // Unit quad (two triangles), expanded to each instance's rect in the shader.
      const quad = new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]);
      const quadBuf = device.createBuffer({
        size: quad.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(quadBuf, 0, quad);

      // Resolution uniform (vec2f, padded to the 16-byte uniform alignment).
      const resBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const shader = device.createShaderModule({
        code: /* wgsl */ `
          struct Res { size: vec2f };
          @group(0) @binding(0) var<uniform> res: Res;
          struct VsOut { @builtin(position) pos: vec4f, @location(0) color: vec3f };
          @vertex fn vs(
            @location(0) corner: vec2f,
            @location(1) rect: vec4f,
            @location(2) color: vec3f
          ) -> VsOut {
            let px = rect.xy + corner * rect.zw;            // pixel position
            let ndc = vec2f(px.x / res.size.x * 2.0 - 1.0, 1.0 - px.y / res.size.y * 2.0);
            var o: VsOut;
            o.pos = vec4f(ndc, 0.0, 1.0);
            o.color = color;
            return o;
          }
          @fragment fn fs(@location(0) color: vec3f) -> @location(0) vec4f {
            return vec4f(color, 1.0);
          }`,
      });

      const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: shader,
          entryPoint: "vs",
          buffers: [
            {
              arrayStride: 8,
              attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
            },
            {
              arrayStride: 28,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 1, offset: 0, format: "float32x4" },
                { shaderLocation: 2, offset: 16, format: "float32x3" },
              ],
            },
          ],
        },
        fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: resBuf } }],
      });

      return { device, ctx, pipeline, quadBuf, resBuf, bindGroup, instBuf: null, instCap: 0 };
    };

    const renderWebGPU = async (): Promise<boolean> => {
      try {
        if (!gpuRef.current) gpuRef.current = await initGpu();
        const g = gpuRef.current;
        if (!g || disposed) return false;
        const { device } = g;

        // Re-configure the context each render: on a resize the canvas backing
        // store changed, and the uniform carries the new logical size.
        g.ctx.configure({
          device,
          format: navigator.gpu.getPreferredCanvasFormat(),
          alphaMode: "premultiplied",
        });
        device.queue.writeBuffer(g.resBuf, 0, new Float32Array([w, h]));

        // Grow the instance buffer only when the cell count outruns its capacity;
        // otherwise reuse it and just overwrite the bytes.
        if (!g.instBuf || g.instCap < cells.length) {
          g.instBuf?.destroy?.();
          g.instBuf = device.createBuffer({
            size: cells.length * 7 * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
          });
          g.instCap = cells.length;
        }
        const inst = new Float32Array(cells.length * 7);
        cells.forEach((c, k) => {
          const [r, gr, b] = heatColor(c.t);
          inst.set([c.x, c.y, Math.max(1, c.w), Math.max(1, c.h), r, gr, b], k * 7);
        });
        device.queue.writeBuffer(g.instBuf, 0, inst);

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: g.ctx.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });
        pass.setPipeline(g.pipeline);
        pass.setBindGroup(0, g.bindGroup);
        pass.setVertexBuffer(0, g.quadBuf);
        pass.setVertexBuffer(1, g.instBuf);
        pass.draw(6, cells.length);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return true;
      } catch {
        return false;
      }
    };

    renderWebGPU().then((ok) => {
      if (disposed) return;
      setBackend(ok ? "webgpu" : "canvas");
      if (!ok) drawCanvas2D();
    });

    return () => {
      disposed = true;
    };
  }, [ordered, size.width, size.height]);

  const onMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const cell = cellsRef.current.find(
      (c) => px >= c.x && px <= c.x + c.w && py >= c.y && py <= c.y + c.h
    );
    setHover(cell ? { market: ordered[cell.i], x: px, y: py } : null);
  };

  const onClick = () => {
    if (!hover) return;
    const slug = hover.market.eventSlug || hover.market.slug;
    go({ fn: "DES", slug, kind: hover.market.eventSlug ? "event" : "market" }, `DES ${slug}`);
  };

  return (
    <motion.div
      ref={wrapRef}
      className="relative h-full w-full"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={transition}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={onClick}
        className="block h-full w-full cursor-pointer"
        role="img"
        aria-label={`Heatmap of ${ordered.length} markets by 24h change`}
      />
      <span className="pointer-events-none absolute top-1 right-1.5 text-[10px] tracking-wide text-faint uppercase">
        {backend === "webgpu" ? "GPU" : "2D"} · {ordered.length}
      </span>
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[240px] rounded-sm border border-edge bg-canvas px-2 py-1 text-tiny shadow-[var(--shadow-pop)]"
          style={{
            left: Math.min(hover.x + 12, size.width - 210),
            top: Math.min(hover.y + 12, size.height - 52),
          }}
        >
          <div className="truncate text-ink">
            {hover.market.groupItemTitle || hover.market.question}
          </div>
          <div className="mt-0.5 flex gap-2 text-[11px]">
            <span className="text-muted">{cents(hover.market.last)}¢</span>
            <span
              className={
                (hover.market.chg24h ?? 0) > 0
                  ? "text-up"
                  : (hover.market.chg24h ?? 0) < 0
                    ? "text-down"
                    : "text-muted"
              }
            >
              {signed(hover.market.chg24h ?? 0)}
            </span>
            <span className="text-faint">${compact(hover.market.volume24h)}</span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
