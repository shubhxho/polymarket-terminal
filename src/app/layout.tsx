import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import Link from "next/link";
import { Suspense } from "react";
import { Clock } from "@/components/clock";
import { SearchFocuser } from "@/components/search-focuser";
import { Terminal } from "@/components/terminal";
import { Ticker } from "@/components/ticker";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "POLYMARKET TERMINAL",
    template: "%s · POLYMARKET TERMINAL",
  },
  description:
    "Real-time prediction market terminal. Prices, volume and odds from Polymarket.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <script
          // Restore the saved phosphor theme before first paint to avoid a color flash
          // biome-ignore lint/security/noDangerouslySetInnerHtml: static inline snippet, no user input
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("pm-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}`,
          }}
        />
        <header className="sticky top-0 z-40 border-b border-edge bg-panel/95 backdrop-blur-sm panel-lit">
          <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-2">
            <Link href="/" className="group flex items-baseline gap-2">
              <span className="bg-accent px-1.5 py-0.5 text-xs font-bold text-black shadow-[0_0_14px_var(--accent-dim)]">
                PM
              </span>
              <span className="text-sm font-bold tracking-widest text-foreground group-hover:text-accent group-hover:glow-soft">
                POLYMARKET TERMINAL
              </span>
              <span className="cursor-blink text-accent">▊</span>
            </Link>
            <div className="flex items-center gap-3 text-xs text-muted sm:gap-4">
              <span className="hidden items-center gap-1.5 text-[11px] text-muted/70 lg:flex">
                <span className="text-accent/70">SYS</span>
                <span className="text-accent">NOMINAL</span>
              </span>
              <span className="hidden h-3 w-px bg-edge lg:block" />
              <span className="hidden items-center gap-2 sm:flex">
                <span className="relative inline-flex h-1.5 w-1.5">
                  <span className="ping-ring absolute inline-flex h-full w-full" />
                  <span className="relative inline-block h-1.5 w-1.5 rounded-full bg-accent" />
                </span>
                <span className="text-accent glow-soft">LIVE</span>
              </span>
              <span className="hidden text-foreground/80 sm:block">
                <Clock />
              </span>
            </div>
          </div>
          <div className="rule-glow h-px" />
        </header>
        <SearchFocuser />
        <Suspense
          fallback={
            <div className="h-[30px] border-b border-edge bg-panel-raised" />
          }
        >
          <Ticker />
        </Suspense>
        <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-4">
          {children}
        </div>
        <footer className="border-t border-edge bg-panel panel-lit">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-2 text-[11px] text-muted">
            <span className="flex items-center gap-1.5">
              <span className="live-dot inline-block h-1 w-1 rounded-full bg-accent/70" />
              DATA: GAMMA-API.POLYMARKET.COM · CLOB.POLYMARKET.COM
            </span>
            <span className="hidden md:block">
              <kbd className="border border-edge bg-panel-raised px-1">`</kbd>{" "}
              TERMINAL ·{" "}
              <kbd className="border border-edge bg-panel-raised px-1">/</kbd>{" "}
              SEARCH
            </span>
            <span>NOT FINANCIAL ADVICE · PRICES = IMPLIED PROBABILITY</span>
          </div>
        </footer>
        <Suspense fallback={null}>
          <Terminal />
        </Suspense>
      </body>
    </html>
  );
}
