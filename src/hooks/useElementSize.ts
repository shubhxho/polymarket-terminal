"use client";

import { useEffect, useRef, useState } from "react";

/** Tracks a element's content box. Charts need real pixels, not percentages,
 *  to place ticks and labels on integer coordinates. */
export function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setSize((prev) =>
        // Sub-pixel jitter would otherwise re-render the chart continuously.
        Math.abs(prev.width - box.width) < 1 && Math.abs(prev.height - box.height) < 1
          ? prev
          : { width: box.width, height: box.height }
      );
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, size] as const;
}
