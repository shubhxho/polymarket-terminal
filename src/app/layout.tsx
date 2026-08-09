import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Providers } from "@/components/Providers";
import "./globals.css";

/**
 * Inter carries the interface — chrome, labels and the data tables alike. Its
 * tabular figures hold a price column steady, which is the one thing this UI
 * cannot compromise on, and it sets far calmer than monospace at 12px.
 */
const ui = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

/**
 * Monospace is reserved for machine identifiers: the command line, token ids,
 * condition hashes and wallet addresses. Self-hosted by `next/font`, so there
 * is no FOUT and no layout shift as the grid paints.
 */
const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
  variable: "--font-terminal",
});

const TITLE = "PMT · Polymarket Terminal";
const DESCRIPTION =
  "A trading terminal for Polymarket — live order books, depth, time & sales, movers, signals and basket arbitrage.";

// Absolute base for og:image / twitter:image. Vercel provides the canonical
// production host at build; fall back to localhost in dev.
const SITE_URL = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : "http://localhost:3000";

// og:image / twitter:image are wired automatically from app/opengraph-image.tsx
// and app/twitter-image.tsx; here we set the card type, title and description.
export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  applicationName: "PMT",
  openGraph: { title: TITLE, description: DESCRIPTION, type: "website", siteName: "PMT" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`h-full ${ui.variable} ${mono.variable}`}>
      <body className="h-full bg-canvas text-ink">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
