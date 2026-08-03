"use client";

import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { Chip, Segmented } from "@/components/ui/Panel";
import { useWallet } from "@/hooks/useWallet";
import {
  buildOrder,
  clearCreds,
  ensureApiCreds,
  isAddress,
  loadCreds,
  loadFunder,
  roundToTick,
  saveCreds,
  saveFunder,
  SIG_TYPE,
  submitOrder,
  type OrderType,
  type SignatureType,
  type Side,
} from "@/lib/clob";
import { cents, usd } from "@/lib/format";
import { tapScale } from "@/lib/motion";
import type { OrderBook } from "@/lib/types";

/** Polymarket rejects anything under a dollar of notional. */
const MIN_NOTIONAL = 1;

/**
 * In-panel order ticket for the selected outcome.
 *
 * Price is entered in cents (0–100) to match how the rest of the terminal
 * quotes probability; internally it snaps to the market tick and rides at
 * 0..1. Two-stage by design: with **LIVE** off, the primary button only builds
 * and shows the exact order payload (no wallet touched); flip LIVE on and it
 * mints/loads the API key, signs the order, and posts it for real. That gate
 * keeps a mis-typed size from turning into a live fill on the first click.
 */
export function OrderTicket({
  tokenId,
  outcomeLabel,
  markPrice,
  tickSize,
  negRisk,
  acceptingOrders,
  book,
}: {
  tokenId: string | undefined;
  outcomeLabel: string | undefined;
  markPrice: number | undefined;
  tickSize: number;
  negRisk: boolean;
  acceptingOrders: boolean;
  book: OrderBook | undefined;
}) {
  const { status, address, onPolygon, switchToPolygon, signTypedData, connect } = useWallet();
  const { toast } = useTerminal();

  const [side, setSide] = useState<Side>("BUY");
  const [orderType, setOrderType] = useState<OrderType>("GTC");
  /** Minutes until a GTD order expires; ignored for GTC/FOK. */
  const [gtdMins, setGtdMins] = useState("60");
  const [priceCents, setPriceCents] = useState("");
  const [size, setSize] = useState("");
  const [live, setLive] = useState(false);
  const [funder, setFunder] = useState(() => loadFunder());
  const [sigType, setSigType] = useState<SignatureType>(SIG_TYPE.POLY_GNOSIS_SAFE);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);

  const bestBid = book?.bids[0]?.price;
  const bestAsk = book?.asks[0]?.price;

  const price = useMemo(() => {
    const c = parseFloat(priceCents);
    if (!Number.isFinite(c)) return NaN;
    return roundToTick(c / 100, tickSize);
  }, [priceCents, tickSize]);

  const shares = parseFloat(size);
  const notional = Number.isFinite(price) && Number.isFinite(shares) ? price * shares : NaN;

  const gtdMinutes = parseFloat(gtdMins);

  const connected = status === "connected" && !!address;

  const problem = useMemo(() => {
    if (!tokenId) return "no outcome selected";
    if (!acceptingOrders) return "market is closed to orders";
    if (!Number.isFinite(price) || price <= 0 || price >= 1) return "price must be 1–99¢";
    if (!Number.isFinite(shares) || shares <= 0) return "enter a share size";
    if (!Number.isFinite(notional) || notional < MIN_NOTIONAL) return "min order is $1";
    if (orderType === "GTD" && !(gtdMinutes > 0)) return "set a GTD expiry in minutes";
    return null;
  }, [tokenId, acceptingOrders, price, shares, notional, orderType, gtdMinutes]);

  /** Requirements that only matter once you actually go live. */
  const liveProblem = useMemo(() => {
    if (!connected) return "connect a wallet";
    if (!isAddress(funder)) return "set your proxy (funder) address";
    return null;
  }, [connected, funder]);

  const prefill = (p: number | undefined) => {
    if (p === undefined) return;
    setPriceCents(cents(p, tickSize >= 0.01 ? 0 : 1));
  };

  const draft = () => ({
    side,
    tokenId: tokenId ?? "",
    price,
    size: shares,
    funder: funder.trim(),
    signer: address ?? "",
    negRisk,
    signatureType: sigType,
    // GTD only: unix seconds when the resting order should expire. Left off for
    // GTC/FOK so buildOrder emits "0".
    expiration:
      orderType === "GTD" && gtdMinutes > 0
        ? Math.floor(Date.now() / 1000) + Math.round(gtdMinutes * 60)
        : undefined,
  });

  const doPreview = () => {
    if (problem) {
      toast(problem, "warn");
      return;
    }
    const built = buildOrder(draft());
    setPreview(JSON.stringify(built.post, null, 2));
    setLastResult(null);
  };

  const doSubmit = async () => {
    if (problem) {
      toast(problem, "warn");
      return;
    }
    if (liveProblem) {
      toast(liveProblem, "warn");
      return;
    }
    const verb = side === "BUY" ? "Buy" : "Sell";
    const confirmed = window.confirm(
      `${verb} ${shares} × "${outcomeLabel ?? "outcome"}" @ ${cents(price)}¢\n` +
        `Notional ${usd(notional)} · ${orderType}\n\n` +
        `This signs and submits a REAL order to Polymarket. Continue?`
    );
    if (!confirmed) return;

    setBusy(true);
    setLastResult(null);
    try {
      if (address) saveFunder(funder.trim());
      const built = buildOrder(draft());
      setPreview(JSON.stringify(built.post, null, 2));

      let creds = loadCreds(address as string);
      if (!creds) {
        creds = await ensureApiCreds(signTypedData, address as string);
        saveCreds(address as string, creds);
      }
      const res = await submitOrder(signTypedData, address as string, creds, built, orderType);
      const label = `${res.status ?? "submitted"}${res.orderId ? ` · ${res.orderId.slice(0, 10)}…` : ""}`;
      setLastResult(label);
      toast(`order ${label}`, "info");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // A stale/rotated key looks like an auth failure — drop it so the next
      // attempt re-mints rather than looping on the dead credential.
      if (/api ?key|unauthor|401|403|invalid.*sig/i.test(msg)) clearCreds(address as string);
      setLastResult(`failed · ${msg}`);
      toast(msg, "error");
    } finally {
      setBusy(false);
    }
  };

  const buyActive = side === "BUY";

  return (
    <div className="flex flex-col gap-1.5 text-tiny">
      {/* Side + order type */}
      <div className="flex items-center justify-between gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-edge">
          {(["BUY", "SELL"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSide(s)}
              className={`px-3 py-[3px] text-[11px] font-bold tracking-wide ${
                side === s
                  ? s === "BUY"
                    ? "bg-up/15 text-up"
                    : "bg-down/15 text-down"
                  : "text-muted hover:text-ink"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <Segmented
          size="xs"
          value={orderType}
          onChange={(v) => setOrderType(v as OrderType)}
          options={[
            { value: "GTC", label: "GTC", title: "Good-til-cancelled limit" },
            { value: "FOK", label: "FOK", title: "Fill-or-kill (marketable)" },
            { value: "GTD", label: "GTD", title: "Good-til-date — expires after the set minutes" },
          ]}
        />
      </div>

      {/* GTD expiry — only when the order type actually needs one. */}
      {orderType === "GTD" ? (
        <label className="flex items-center gap-2">
          <span className="w-[38px] shrink-0 text-[10px] tracking-wide text-muted uppercase">
            Expiry
          </span>
          <input
            inputMode="numeric"
            value={gtdMins}
            onChange={(e) => setGtdMins(e.target.value)}
            className="w-[70px] border border-edge bg-surface-2 px-1.5 py-[3px] text-tiny text-ink"
            aria-label="GTD expiry in minutes"
          />
          <span className="text-[10px] text-faint">
            minutes
            {gtdMinutes > 0
              ? ` · ~${new Date(Date.now() + gtdMinutes * 60000).toLocaleTimeString()}`
              : ""}
          </span>
        </label>
      ) : null}

      {/* Price */}
      <label className="flex items-center gap-2">
        <span className="w-[38px] shrink-0 text-[10px] tracking-wide text-muted uppercase">
          Price
        </span>
        <input
          value={priceCents}
          onChange={(e) => setPriceCents(e.target.value)}
          onKeyDown={(e) => e.stopPropagation()}
          inputMode="decimal"
          placeholder="cents"
          className="min-w-0 flex-1 border border-edge bg-surface-2 px-1.5 py-[3px] text-ink placeholder:text-faint"
        />
        <span className="flex shrink-0 gap-0.5">
          <QuickPrice
            label="BID"
            onClick={() => prefill(bestBid)}
            disabled={bestBid === undefined}
          />
          <QuickPrice
            label="ASK"
            onClick={() => prefill(bestAsk)}
            disabled={bestAsk === undefined}
          />
          <QuickPrice
            label="MARK"
            onClick={() => prefill(markPrice)}
            disabled={markPrice === undefined}
          />
        </span>
      </label>

      {/* Size */}
      <label className="flex items-center gap-2">
        <span className="w-[38px] shrink-0 text-[10px] tracking-wide text-muted uppercase">
          Shares
        </span>
        <input
          value={size}
          onChange={(e) => setSize(e.target.value)}
          onKeyDown={(e) => e.stopPropagation()}
          inputMode="decimal"
          placeholder="qty"
          className="min-w-0 flex-1 border border-edge bg-surface-2 px-1.5 py-[3px] text-ink placeholder:text-faint"
        />
        <span className="w-[92px] shrink-0 text-right text-[10px] text-muted">
          {Number.isFinite(notional) ? (
            <>
              cost <span className="text-ink">{usd(notional)}</span>
            </>
          ) : (
            "cost --"
          )}
        </span>
      </label>

      {/* Quick-size by target cost. Traders size in dollars; convert to shares
          at the working price the way every exchange offers amount presets.
          Disabled until a price is set, since shares = cost / price. */}
      <div className="flex items-center gap-2">
        <span className="w-[38px] shrink-0" aria-hidden />
        <span className="flex shrink-0 gap-0.5">
          {[25, 50, 100, 250].map((dollars) => (
            <QuickPrice
              key={dollars}
              label={`$${dollars}`}
              onClick={() => setSize(String(Math.max(1, Math.round(dollars / price))))}
              disabled={!(price > 0)}
            />
          ))}
        </span>
        <span className="text-[9px] text-faint">target cost</span>
      </div>

      {/* Advanced: funder + signature type */}
      <button
        onClick={() => setAdvanced((a) => !a)}
        className="self-start text-[10px] tracking-wide text-accent-weak uppercase hover:text-accent"
      >
        {advanced ? "▾ account" : "▸ account"}
      </button>
      {advanced ? (
        <div className="flex flex-col gap-1 border-l border-edge pl-2">
          <label className="flex items-center gap-2">
            <span className="w-[46px] shrink-0 text-[10px] text-muted">Funder</span>
            <input
              value={funder}
              onChange={(e) => setFunder(e.target.value)}
              onKeyDown={(e) => e.stopPropagation()}
              placeholder="0x… proxy wallet that holds your USDC"
              className="min-w-0 flex-1 border border-edge bg-surface-2 px-1.5 py-[2px] font-mono text-[10px] text-ink placeholder:text-faint"
            />
          </label>
          <div className="flex items-center gap-2">
            <span className="w-[46px] shrink-0 text-[10px] text-muted">Sig</span>
            <Segmented
              size="xs"
              value={String(sigType)}
              onChange={(v) => setSigType(Number(v) as SignatureType)}
              options={[
                {
                  value: String(SIG_TYPE.POLY_GNOSIS_SAFE),
                  label: "Safe",
                  title: "Browser-wallet proxy (Phantom/MetaMask)",
                },
                { value: String(SIG_TYPE.POLY_PROXY), label: "Email", title: "Magic/email proxy" },
                {
                  value: String(SIG_TYPE.EOA),
                  label: "EOA",
                  title: "Trade directly from the connected address",
                },
              ]}
            />
          </div>
          <p className="text-[9px] leading-snug text-faint">
            Funder = the wallet that holds your Polymarket balance (your deposit address on
            polymarket.com), not necessarily the connected account. Requires USDC & CTF allowances
            already set.
          </p>
        </div>
      ) : null}

      {/* LIVE gate */}
      <div className="mt-0.5 flex items-center justify-between gap-2 border-t border-edge pt-1.5">
        <button
          onClick={() => setLive((l) => !l)}
          className={`inline-flex items-center gap-1.5 text-[10px] font-bold tracking-wide uppercase ${
            live ? "text-down" : "text-muted hover:text-ink"
          }`}
        >
          <span
            className={`inline-block h-[10px] w-[18px] rounded-full border transition-colors ${
              live ? "border-down bg-down/30" : "border-edge bg-surface-2"
            }`}
          >
            <span
              className={`block h-[8px] w-[8px] rounded-full transition-transform ${
                live ? "translate-x-[8px] bg-down" : "translate-x-0 bg-muted"
              }`}
            />
          </span>
          {live ? "LIVE" : "SAFE"}
        </button>
        {!onPolygon && connected ? (
          <Chip tone="warn" onClick={() => void switchToPolygon()}>
            switch to Polygon
          </Chip>
        ) : null}
      </div>

      {/* Primary action */}
      {!connected ? (
        <motion.button
          whileTap={tapScale}
          onClick={() => void connect()}
          className="w-full border border-accent-weak py-[5px] text-[11px] font-bold tracking-wide text-accent hover:bg-accent hover:text-canvas"
        >
          CONNECT WALLET
        </motion.button>
      ) : live ? (
        <motion.button
          whileTap={tapScale}
          onClick={() => void doSubmit()}
          disabled={busy || !!problem}
          className={`w-full border py-[5px] text-[11px] font-bold tracking-wide disabled:opacity-40 ${
            buyActive
              ? "border-up text-up hover:bg-up hover:text-canvas"
              : "border-down text-down hover:bg-down hover:text-canvas"
          }`}
        >
          {busy ? "SIGNING…" : `SIGN & SUBMIT ${side}`}
        </motion.button>
      ) : (
        <motion.button
          whileTap={tapScale}
          onClick={doPreview}
          disabled={!!problem}
          className="w-full border border-edge-strong py-[5px] text-[11px] font-bold tracking-wide text-ink hover:border-accent-weak hover:text-accent disabled:opacity-40"
        >
          PREVIEW ORDER
        </motion.button>
      )}

      {problem ? (
        <div className="text-[10px] text-warn">{problem}</div>
      ) : liveProblem && live ? (
        <div className="text-[10px] text-warn">{liveProblem}</div>
      ) : null}

      {lastResult ? (
        <div
          className={`truncate text-[10px] ${lastResult.startsWith("failed") ? "text-down" : "text-up"}`}
          title={lastResult}
        >
          {lastResult}
        </div>
      ) : null}

      {preview ? (
        <details className="mt-0.5">
          <summary className="cursor-pointer text-[10px] tracking-wide text-accent-weak uppercase">
            signed payload
          </summary>
          <pre className="mt-1 max-h-[150px] overflow-auto border border-edge bg-surface-2 p-1.5 text-[9px] leading-snug text-muted">
            {preview}
          </pre>
        </details>
      ) : null}
    </div>
  );
}

function QuickPrice({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="border border-edge px-1 py-[2px] text-[9px] tracking-wide text-muted hover:border-accent-weak hover:text-accent disabled:opacity-30"
    >
      {label}
    </button>
  );
}
