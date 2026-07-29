"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

/**
 * Phantom wallet, on the EVM side.
 *
 * Polymarket settles on Polygon, so the address that matters here is a 0x
 * account — exactly what the PORT screen already consumes. Phantom is
 * multichain and injects an EIP-1193 provider for EVM at `window.phantom.ethereum`;
 * we speak to that rather than the Solana object, so a connect resolves to a
 * wallet the blotter can quote without any translation.
 */

/** Polygon mainnet — the chain Polymarket clears on. */
export const POLYGON_HEX = "0x89";
const POLYGON_PARAMS = {
  chainId: POLYGON_HEX,
  chainName: "Polygon",
  nativeCurrency: { name: "POL", symbol: "POL", decimals: 18 },
  rpcUrls: ["https://polygon-rpc.com"],
  blockExplorerUrls: ["https://polygonscan.com"],
};

/** Minimal EIP-1193 surface — all we touch of the injected provider. */
type Eip1193 = {
  isPhantom?: boolean;
  request: (args: { method: string; params?: unknown[] | object }) => Promise<unknown>;
  on?: (event: string, handler: (...args: unknown[]) => void) => void;
  removeListener?: (event: string, handler: (...args: unknown[]) => void) => void;
};

type PhantomWindow = Window & {
  phantom?: { ethereum?: Eip1193 };
  ethereum?: Eip1193;
};

/** Phantom's own EVM provider, never a different wallet's injection. */
function getProvider(): Eip1193 | null {
  if (typeof window === "undefined") return null;
  const w = window as PhantomWindow;
  const phantom = w.phantom?.ethereum;
  if (phantom?.isPhantom) return phantom;
  // Some builds only mirror onto window.ethereum; accept it iff it's Phantom,
  // so we never hijack MetaMask sitting in the same slot.
  if (w.ethereum?.isPhantom) return w.ethereum;
  return null;
}

export type WalletStatus = "unavailable" | "disconnected" | "connecting" | "connected";

type WalletCtx = {
  status: WalletStatus;
  /** Checksum-agnostic lowercase 0x address, or null when disconnected. */
  address: string | null;
  /** Hex chain id of the connected wallet, or null. */
  chainId: string | null;
  onPolygon: boolean;
  /** True once we know whether Phantom is installed (avoids a flash of "install"). */
  ready: boolean;
  /** Whether `addr` is the connected wallet — owns the address-casing rule. */
  isMe: (addr: string) => boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
  switchToPolygon: () => Promise<void>;
};

const Ctx = createContext<WalletCtx | null>(null);

/** Remembers that the user opted in, so a reload silently re-links. */
const OPT_IN_KEY = "pmt.wallet.optin";

export function WalletProvider({
  children,
  onConnect,
  onError,
}: {
  children: ReactNode;
  onConnect?: (address: string) => void;
  onError?: (message: string) => void;
}) {
  // One tri-state instead of two booleans: "checking" until detection resolves,
  // then "present" or "absent". `ready` and the "unavailable" status both derive
  // from it, so they can never disagree.
  const [providerState, setProviderState] = useState<"checking" | "present" | "absent">("checking");
  const [address, setAddress] = useState<string | null>(null);
  const [chainId, setChainId] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  // Callbacks are read through a ref so the connect/effect logic never has to
  // list them as deps and re-subscribe every render.
  const cbRef = useRef({ onConnect, onError });
  useEffect(() => {
    cbRef.current = { onConnect, onError };
  }, [onConnect, onError]);

  const applyAccounts = useCallback((accts: unknown) => {
    const list = Array.isArray(accts) ? (accts as string[]) : [];
    setAddress(list.length ? list[0].toLowerCase() : null);
  }, []);

  // Detect Phantom and, if the user previously connected, re-link silently via
  // eth_accounts (which never prompts). Also wires the live event handlers.
  useEffect(() => {
    const provider = getProvider();
    let cancelled = false;

    const onAccounts = (...args: unknown[]) => applyAccounts(args[0]);
    const onChain = (...args: unknown[]) =>
      setChainId(typeof args[0] === "string" ? (args[0] as string) : null);
    const onDisconnect = () => setAddress(null);

    if (provider) {
      provider.on?.("accountsChanged", onAccounts);
      provider.on?.("chainChanged", onChain);
      provider.on?.("disconnect", onDisconnect);
    }

    // All initial state lands inside this async pass, never synchronously in
    // the effect body — the detection is a read of an external system, and its
    // result should arrive the same way a socket's first frame does.
    (async () => {
      if (cancelled) return;
      if (!provider) {
        setProviderState("absent");
        return;
      }
      try {
        const optedIn = window.localStorage.getItem(OPT_IN_KEY) === "1";
        const chain = (await provider.request({ method: "eth_chainId" })) as string;
        if (!cancelled) setChainId(chain);
        if (optedIn) {
          const accts = await provider.request({ method: "eth_accounts" });
          if (!cancelled) applyAccounts(accts);
        }
      } catch {
        // A cold provider with no session — nothing to restore.
      } finally {
        if (!cancelled) setProviderState("present");
      }
    })();

    return () => {
      cancelled = true;
      provider?.removeListener?.("accountsChanged", onAccounts);
      provider?.removeListener?.("chainChanged", onChain);
      provider?.removeListener?.("disconnect", onDisconnect);
    };
  }, [applyAccounts]);

  // Announce a fresh connection exactly once per address, not on every re-link.
  const announced = useRef<string | null>(null);
  useEffect(() => {
    if (address && announced.current !== address) {
      announced.current = address;
      cbRef.current.onConnect?.(address);
    }
    if (!address) announced.current = null;
  }, [address]);

  const connect = useCallback(async () => {
    const provider = getProvider();
    if (!provider) {
      // No Phantom — send them to install it rather than failing silently.
      window.open("https://phantom.com/download", "_blank", "noopener,noreferrer");
      cbRef.current.onError?.("Phantom not detected — install it to connect");
      return;
    }
    setConnecting(true);
    try {
      const accts = await provider.request({ method: "eth_requestAccounts" });
      applyAccounts(accts);
      const chain = (await provider.request({ method: "eth_chainId" })) as string;
      setChainId(chain);
      window.localStorage.setItem(OPT_IN_KEY, "1");
    } catch (err) {
      const code = (err as { code?: number })?.code;
      // 4001 is the user waving the prompt away — not worth an error toast.
      if (code !== 4001) {
        cbRef.current.onError?.(
          err instanceof Error ? err.message : "could not connect to Phantom"
        );
      }
    } finally {
      setConnecting(false);
    }
  }, [applyAccounts]);

  const disconnect = useCallback(() => {
    // Phantom has no programmatic revoke for EVM; forgetting the opt-in and
    // dropping the address is the honest local equivalent.
    window.localStorage.removeItem(OPT_IN_KEY);
    setAddress(null);
  }, []);

  const switchToPolygon = useCallback(async () => {
    const provider = getProvider();
    if (!provider) return;
    try {
      await provider.request({
        method: "wallet_switchEthereumChain",
        params: [{ chainId: POLYGON_HEX }],
      });
    } catch (err) {
      // 4902: chain unknown to the wallet — offer to add it, then it's switched.
      if ((err as { code?: number })?.code === 4902) {
        try {
          await provider.request({
            method: "wallet_addEthereumChain",
            params: [POLYGON_PARAMS],
          });
        } catch {
          cbRef.current.onError?.("could not add Polygon to Phantom");
        }
      }
    }
  }, []);

  const isMe = useCallback(
    (addr: string) => address !== null && address === addr.toLowerCase(),
    [address]
  );

  const status: WalletStatus =
    providerState === "absent"
      ? "unavailable"
      : connecting
        ? "connecting"
        : address
          ? "connected"
          : "disconnected";

  const value = useMemo<WalletCtx>(
    () => ({
      status,
      address,
      chainId,
      onPolygon: chainId === POLYGON_HEX,
      ready: providerState !== "checking",
      isMe,
      connect,
      disconnect,
      switchToPolygon,
    }),
    [status, address, chainId, providerState, isMe, connect, disconnect, switchToPolygon]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useWallet(): WalletCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useWallet must be used inside <WalletProvider>");
  return ctx;
}

/** Short 0x1234…abcd form for chrome; full string stays in the DOM title. */
export function shortAddress(addr: string): string {
  return addr.length > 10 ? `${addr.slice(0, 6)}…${addr.slice(-4)}` : addr;
}
