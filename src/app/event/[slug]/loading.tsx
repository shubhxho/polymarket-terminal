export default function Loading() {
  return (
    <main className="flex flex-1 flex-col gap-3">
      <div className="text-xs text-muted">
        <span>~/MARKETS</span>
        <span>
          {" "}
          / <span className="cursor-blink">▊</span>
        </span>
      </div>

      {/* Header skeleton */}
      <div className="flex flex-wrap items-start justify-between gap-3 border border-edge bg-panel p-3 panel-lit">
        <div className="flex items-center gap-3">
          <div className="shimmer h-10 w-10 shrink-0 rounded-sm bg-panel-raised" />
          <div className="flex flex-col gap-1.5">
            <div className="shimmer h-4 w-64 rounded-sm bg-panel-raised" />
            <div className="shimmer h-3 w-40 rounded-sm bg-panel-raised" />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-px bg-edge sm:grid-cols-4">
          {["24H VOL", "TOTAL VOL", "LIQUIDITY", "OPEN INT"].map((l) => (
            <div key={l} className="bg-panel px-3 py-1.5 panel-lit">
              <div className="text-[10px] tracking-widest text-muted">{l}</div>
              <div className="shimmer mt-0.5 h-4 w-16 rounded-sm bg-panel-raised" />
            </div>
          ))}
        </div>
      </div>

      {/* Chart skeleton */}
      <div className="flex items-center justify-between">
        <div className="h-3 w-32 rounded-sm bg-panel-raised" />
        <div className="flex gap-px border border-edge bg-edge">
          {["1D", "1W", "1M", "3M", "MAX"].map((r) => (
            <span key={r} className="bg-panel px-3 py-1 text-[11px] text-muted/30">
              {r}
            </span>
          ))}
        </div>
      </div>
      <div className="scan-sweep flex h-[300px] items-center justify-center border border-edge bg-panel text-muted panel-lit">
        <span className="text-accent">&gt;</span>&nbsp;RENDERING PRICE HISTORY
        <span className="cursor-blink text-accent">▊</span>
      </div>

      {/* Order book skeleton */}
      <div className="text-xs tracking-widest text-muted">ORDER BOOK</div>
      <div className="border border-edge bg-panel panel-lit">
        {Array.from({ length: 4 }, (_, i) => `book-row-${i}`).map((id) => (
          <div
            key={id}
            className="flex items-center justify-between border-b border-edge px-3 py-2.5 last:border-b-0"
          >
            <div className="shimmer h-3 w-36 rounded-sm bg-panel-raised" />
            <div className="flex gap-4">
              <div className="shimmer h-3 w-10 rounded-sm bg-panel-raised" />
              <div className="shimmer h-3 w-10 rounded-sm bg-panel-raised" />
              <div className="shimmer h-3 w-24 rounded-sm bg-panel-raised" />
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
