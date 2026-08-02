import { describe, expect, it } from "vitest";
import { buildOrder, isAddress, roundToTick, SIG_TYPE, type OrderDraft } from "@/lib/clob";

const EOA = "0x1111111111111111111111111111111111111111";
const FUNDER = "0x2222222222222222222222222222222222222222";

/** The EIP-712 verifying contract a built order will be signed against. */
const addrOf = (b: ReturnType<typeof buildOrder>) =>
  (b.typedData as { domain: { verifyingContract: string } }).domain.verifyingContract;

function draft(over: Partial<OrderDraft> = {}): OrderDraft {
  return {
    side: "BUY",
    tokenId: "123456789",
    price: 0.63,
    size: 100,
    funder: FUNDER,
    signer: EOA,
    negRisk: false,
    signatureType: SIG_TYPE.POLY_GNOSIS_SAFE,
    ...over,
  };
}

describe("roundToTick", () => {
  it("snaps to the grid and strips float dust", () => {
    expect(roundToTick(0.6301, 0.001)).toBe(0.63);
    expect(roundToTick(0.6349, 0.01)).toBe(0.63);
    expect(roundToTick(0.6351, 0.01)).toBe(0.64);
  });

  it("clamps inside (tick, 1 - tick)", () => {
    expect(roundToTick(0, 0.01)).toBe(0.01);
    expect(roundToTick(1, 0.01)).toBe(0.99);
    expect(roundToTick(5, 0.001)).toBe(0.999);
  });
});

describe("isAddress", () => {
  it("accepts a 20-byte hex address and rejects junk", () => {
    expect(isAddress(FUNDER)).toBe(true);
    expect(isAddress("0x123")).toBe(false);
    expect(isAddress("not-an-address")).toBe(false);
  });
});

describe("buildOrder", () => {
  it("BUY puts USDC on maker, shares on taker (6-decimal base units)", () => {
    const { post, notional } = buildOrder(draft({ side: "BUY", price: 0.63, size: 100 }));
    // 100 shares @ 0.63 = $63 notional
    expect(post.makerAmount).toBe("63000000");
    expect(post.takerAmount).toBe("100000000");
    expect(post.side).toBe("BUY");
    expect(notional).toBeCloseTo(63);
  });

  it("SELL flips maker/taker", () => {
    const { post } = buildOrder(draft({ side: "SELL", price: 0.4, size: 50 }));
    // give 50 shares, receive $20
    expect(post.makerAmount).toBe("50000000");
    expect(post.takerAmount).toBe("20000000");
    expect(post.side).toBe("SELL");
  });

  it("maker is the funder, signer is the EOA", () => {
    const { post } = buildOrder(draft());
    expect(post.maker).toBe(FUNDER);
    expect(post.signer).toBe(EOA);
  });

  it("signs the V2 struct: timestamp/metadata/builder present, no V1 fields", () => {
    const built = buildOrder(draft());
    const types = (built.typedData as { types: { Order: { name: string }[] } }).types.Order;
    const names = types.map((t) => t.name);
    // V2 additions
    expect(names).toContain("timestamp");
    expect(names).toContain("metadata");
    expect(names).toContain("builder");
    // V1 fields dropped from the signed struct
    for (const gone of ["taker", "expiration", "nonce", "feeRateBps"]) {
      expect(names).not.toContain(gone);
    }
    // domain version bumped to "2"
    const domain = (built.typedData as { domain: { version: string } }).domain;
    expect(domain.version).toBe("2");
    // timestamp is milliseconds and shared verbatim with the POST body
    const msg = (built.typedData as { message: { timestamp: string } }).message;
    expect(built.post.timestamp).toBe(msg.timestamp);
    expect(Number(msg.timestamp)).toBeGreaterThan(1_700_000_000_000);
    // expiration lives only in the unsigned POST body
    expect(built.post.expiration).toBe("0");
  });

  it("keeps the signed struct and the POST body on the same salt & amounts", () => {
    const built = buildOrder(draft());
    const msg = (
      built.typedData as {
        message: { salt: string; makerAmount: string; takerAmount: string; side: number };
      }
    ).message;
    // salt is stringified in the struct, numeric in the POST — same value
    expect(String(built.post.salt)).toBe(msg.salt);
    expect(built.post.makerAmount).toBe(msg.makerAmount);
    expect(built.post.takerAmount).toBe(msg.takerAmount);
    // side is a uint8 in the struct, a string in the POST
    expect(msg.side).toBe(0);
    expect(built.post.side).toBe("BUY");
    // salt stays inside the safe-integer range
    expect(built.post.salt).toBeLessThan(Number.MAX_SAFE_INTEGER);
  });

  it("routes neg-risk markets to the V2 neg-risk exchange", () => {
    const std = buildOrder(draft({ negRisk: false }));
    const neg = buildOrder(draft({ negRisk: true }));
    expect(addrOf(std)).not.toBe(addrOf(neg));
    expect(addrOf(std)).toBe("0xE111180000d2663C0091e4f400237545B87B996B");
    expect(addrOf(neg)).toBe("0xe2222d279d744050d28e00520010520000310F59");
  });

  it("emits a fresh salt each call", () => {
    const a = buildOrder(draft());
    const b = buildOrder(draft());
    expect(a.post.salt).not.toBe(b.post.salt);
  });
});
