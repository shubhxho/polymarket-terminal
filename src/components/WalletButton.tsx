"use client";

import { memo, useEffect, useRef, useState } from "react";
import { useTerminal } from "@/components/TerminalProvider";
import { shortAddress, useWallet } from "@/hooks/useWallet";
import { cn } from "@/lib/cn";
import { copyToClipboard } from "@/lib/clipboard";

/**
 * Wallet control in the masthead. Disconnected, it's a single quiet "Connect"
 * pill; connected, it becomes the truncated address plus a chain dot that goes
 * amber off Polygon, with a popover for the three things a connected user
 * actually wants — jump to their book, copy the address, or disconnect.
 */
// Memoized: TopBar re-renders every second off its clock, and this component
// takes no props, so without memo its SVG would reconcile once a second forever.
// It still re-renders on wallet context changes, which is all it needs.
export const WalletButton = memo(function WalletButton() {
  const { status, address, onPolygon, ready, connect, disconnect, switchToPolygon } = useWallet();
  const { go, toast } = useTerminal();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // Close the popover on an outside click or Escape — standard menu manners.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Hold the layout until we know Phantom's state, so the pill doesn't flip
  // from "Connect" to an address a frame after paint.
  if (!ready) {
    return <span className="h-[24px] w-[84px] shrink-0 rounded-md border border-edge" />;
  }

  if (status !== "connected" || !address) {
    const connecting = status === "connecting";
    return (
      <button
        onClick={connect}
        disabled={connecting}
        title={
          status === "unavailable"
            ? "Phantom not detected — click to install"
            : "Connect your Phantom wallet"
        }
        className="flex h-[24px] shrink-0 items-center gap-1.5 rounded-md border border-edge px-2 text-tiny font-medium text-ink hover:border-accent-weak hover:text-accent disabled:opacity-60"
      >
        <PhantomMark />
        <span>{connecting ? "Connecting…" : "Connect"}</span>
      </button>
    );
  }

  // Every menu action closes the popover first, then acts — one place, so the
  // items don't each repeat `setOpen(false)` or lean on the comma operator.
  const act = (fn: () => void) => () => {
    setOpen(false);
    fn();
  };

  return (
    <div ref={rootRef} className="relative shrink-0">
      <button
        onClick={() => setOpen((v) => !v)}
        title={address}
        className="flex h-[24px] items-center gap-1.5 rounded-md border border-edge px-2 text-tiny hover:border-edge-strong"
      >
        <span
          className={cn("dot", onPolygon ? "text-up" : "text-warn")}
          title={onPolygon ? "On Polygon" : "Wrong network — switch to Polygon"}
        />
        <span className="mono text-[11px]">{shortAddress(address)}</span>
      </button>

      {open && (
        <div className="absolute right-0 z-50 mt-1 w-[200px] overflow-hidden rounded-md border border-edge bg-canvas py-1 shadow-[var(--shadow-pop)]">
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <PhantomMark />
            <span className="mono truncate text-[11px] text-muted" title={address}>
              {shortAddress(address)}
            </span>
          </div>
          <div className="my-1 h-px bg-edge" />
          {!onPolygon && (
            <MenuItem onClick={act(switchToPolygon)}>
              <span className="text-warn">Switch to Polygon</span>
            </MenuItem>
          )}
          <MenuItem onClick={act(() => go({ fn: "PORT", user: address }, `PORT ${address}`))}>
            View my portfolio
          </MenuItem>
          <MenuItem onClick={act(() => copyToClipboard(address, toast))}>Copy address</MenuItem>
          <MenuItem
            onClick={act(() => {
              disconnect();
              toast("wallet disconnected");
            })}
          >
            <span className="text-down">Disconnect</span>
          </MenuItem>
        </div>
      )}
    </div>
  );
});

function MenuItem({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="block w-full px-3 py-1.5 text-left text-tiny text-ink hover:bg-surface-2"
    >
      {children}
    </button>
  );
}

/** Phantom's ghost, drawn small enough to sit inside a 24px pill. */
function PhantomMark() {
  return (
    <svg viewBox="0 0 128 128" aria-hidden className="h-3.5 w-3.5 shrink-0">
      <rect width="128" height="128" rx="28" fill="#AB9FF2" />
      <path
        fill="#fff"
        d="M110 64.9c0 25.2-20.5 45.6-45.8 45.6-22 0-40.4-15.5-44.8-36.2-.3-1.3-.4-2.7-.4-4 0-3.3 2.7-6 6-6h9.9c2.8 0 5.2 1.9 6 4.6a20.4 20.4 0 0 0 39.3-.2c.8-2.6 3.1-4.4 5.8-4.4H98c3.3 0 6 2.7 6 6z"
        transform="translate(-6 -8)"
      />
      <circle cx="46" cy="62" r="7" fill="#AB9FF2" />
      <circle cx="70" cy="62" r="7" fill="#AB9FF2" />
    </svg>
  );
}
