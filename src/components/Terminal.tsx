"use client";

import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect } from "react";
import type { Summary } from "@/app/api/summary/route";
import { AlertEngine } from "@/components/AlertEngine";
import { CommandBar } from "@/components/CommandBar";
import { Sidebar } from "@/components/Sidebar";
import { TabStrip } from "@/components/TabStrip";
import { TerminalProvider, useTerminal } from "@/components/TerminalProvider";
import { TickerTape } from "@/components/TickerTape";
import { Toasts } from "@/components/Toasts";
import { TopBar } from "@/components/TopBar";
import AlertsScreen from "@/components/screens/AlertsScreen";
import CategoryScreen from "@/components/screens/CategoryScreen";
import DetailScreen from "@/components/screens/DetailScreen";
import HelpScreen from "@/components/screens/HelpScreen";
import MeshScreen from "@/components/screens/MeshScreen";
import MonitorScreen from "@/components/screens/MonitorScreen";
import MoversScreen from "@/components/screens/MoversScreen";
import PortfolioScreen from "@/components/screens/PortfolioScreen";
import SearchScreen from "@/components/screens/SearchScreen";
import SignalsScreen from "@/components/screens/SignalsScreen";
import TapeScreen from "@/components/screens/TapeScreen";
import WatchlistScreen from "@/components/screens/WatchlistScreen";
import { shortAddress, WalletProvider } from "@/hooks/useWallet";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { usePoll } from "@/hooks/usePoll";
import type { Screen } from "@/lib/commands";
import { screenVariants } from "@/lib/motion";

type Theme = "light" | "dark";

export function Terminal() {
  return (
    <TerminalProvider>
      <Shell />
    </TerminalProvider>
  );
}

function Shell() {
  const { screen, toast } = useTerminal();
  // One shared breadth poll feeds both the masthead and the tape; polling it
  // twice would double the upstream load for identical data.
  const summary = usePoll<Summary>("/api/summary", 20000);

  const [theme, setTheme] = useLocalStorage<Theme>("pmt.theme", "light");

  // The palette lives entirely in CSS custom properties keyed off this
  // attribute, so flipping it re-themes every component without a re-render.
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggleTheme = useCallback(
    () => setTheme((t) => (t === "light" ? "dark" : "light")),
    [setTheme]
  );

  return (
    <WalletProvider
      onConnect={(addr) => toast(`wallet connected · ${shortAddress(addr)}`)}
      onError={(msg) => toast(msg, "error")}
    >
      <div className="flex h-full flex-col overflow-hidden bg-canvas">
        <TopBar summary={summary} theme={theme} onToggleTheme={toggleTheme} />
        <CommandBar />
        <TabStrip />

        <div className="flex min-h-0 flex-1">
          <Sidebar />
          {/* `key` remounts on navigation so each screen starts from a clean
              slate instead of inheriting the last one's selection and scroll.
              `AnimatePresence` crossfades the swap; `popLayout` takes the
              outgoing screen out of flow so the two never reflow each other
              mid-flight. */}
          <main className="min-w-0 flex-1 overflow-hidden p-2">
            <AnimatePresence mode="popLayout" initial={false}>
              <motion.div
                key={screenKey(screen)}
                variants={screenVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="h-full min-h-0"
              >
                <Workspace screen={screen} />
              </motion.div>
            </AnimatePresence>
          </main>
        </div>

        <TickerTape summary={summary.data} />
        <Toasts />
        <AlertEngine />
      </div>
    </WalletProvider>
  );
}

function Workspace({ screen }: { screen: Screen }) {
  switch (screen.fn) {
    case "MON":
      return <MonitorScreen />;
    case "SIG":
      return <SignalsScreen />;
    case "MOV":
      return <MoversScreen />;
    case "DES":
      return <DetailScreen slug={screen.slug} kind={screen.kind} />;
    case "SRCH":
      return <SearchScreen q={screen.q} />;
    case "WATCH":
      return <WatchlistScreen />;
    case "TAS":
      return <TapeScreen />;
    case "PORT":
      return <PortfolioScreen user={screen.user} />;
    case "ALRT":
      return <AlertsScreen />;
    case "MESH":
      return <MeshScreen />;
    case "CAT":
      return <CategoryScreen tag={screen.tag} label={screen.label} />;
    case "HELP":
      return <HelpScreen />;
  }
}

/** Stable identity for a screen, including its arguments. */
function screenKey(screen: Screen): string {
  switch (screen.fn) {
    case "DES":
      return `DES:${screen.slug}`;
    case "SRCH":
      return `SRCH:${screen.q}`;
    case "PORT":
      return `PORT:${screen.user}`;
    case "CAT":
      return `CAT:${screen.tag}`;
    default:
      return screen.fn;
  }
}
