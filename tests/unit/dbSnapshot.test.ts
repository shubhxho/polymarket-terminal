import { describe, expect, test } from "vitest";
import { recordSignalSnapshot, snapshotRow } from "@/db/snapshot";
import { db, dbEnabled } from "@/db/client";
import type { SignalsPayload } from "@/app/api/signals/route";

const market = (id: string, over: Record<string, unknown> = {}) =>
  ({
    market: { id, slug: id, question: `Will ${id}?` },
    heat: 70,
    bias: 40,
    conviction: 50,
    model: { prob: 0.66, direction: "bullish", conviction: 0.32, auc: 0.6502 },
    signals: [],
    stats: {},
    ...over,
  }) as unknown as SignalsPayload["markets"][number];

const payload = (over: Partial<SignalsPayload> = {}): SignalsPayload =>
  ({
    arbs: [{}, {}],
    drifts: [{}],
    markets: [market("alpha"), market("beta", { heat: 40, model: undefined })],
    stats: {
      scanned: 180,
      flagged: 12,
      bullish: 7,
      bearish: 3,
      deepScanned: 40,
      byKind: {},
      blockNotional: 250_000,
      modeled: 10,
      modelConfirms: 6,
      modelConflicts: 2,
    },
    ...over,
  }) as unknown as SignalsPayload;

describe("snapshotRow", () => {
  test("flattens a scan payload into the insert row", () => {
    const row = snapshotRow(payload());
    expect(row.scanned).toBe(180);
    expect(row.flagged).toBe(12);
    expect(row.modeled).toBe(10);
    expect(row.confirms).toBe(6);
    expect(row.conflicts).toBe(2);
    expect(row.bullish).toBe(7);
    expect(row.bearish).toBe(3);
    expect(row.arbs).toBe(2);
    expect(row.drifts).toBe(1);
    expect(row.blockNotional).toBe(250_000);
    expect(row.topMarketId).toBe("alpha");
    expect(row.topHeat).toBe(70);
  });

  test("keeps a compact per-market blob, model prob null when unmodeled", () => {
    const row = snapshotRow(payload());
    const markets = row.markets as { id: string; prob: number | null }[];
    expect(markets).toHaveLength(2);
    expect(markets[0]).toMatchObject({ id: "alpha", heat: 70, bias: 40, prob: 0.66 });
    expect(markets[1].prob).toBeNull();
  });

  test("caps the market blob at 30 rows", () => {
    const many = Array.from({ length: 50 }, (_, i) => market(`m${i}`));
    const row = snapshotRow(payload({ markets: many as SignalsPayload["markets"] }));
    expect((row.markets as unknown[]).length).toBe(30);
  });

  test("tolerates an empty scan", () => {
    const row = snapshotRow(
      payload({ markets: [] as unknown as SignalsPayload["markets"], arbs: [], drifts: [] })
    );
    expect(row.topMarketId).toBeNull();
    expect(row.topHeat).toBeNull();
    expect(row.markets).toEqual([]);
  });
});

describe("db graceful degradation", () => {
  test("dbEnabled() reflects DATABASE_URL, db() matches", () => {
    const has = Boolean(process.env.DATABASE_URL);
    expect(dbEnabled()).toBe(has);
    if (!has) expect(db()).toBeNull();
  });

  test("recordSignalSnapshot resolves without throwing even with no DB", async () => {
    await expect(recordSignalSnapshot(payload())).resolves.toBeUndefined();
  });
});
