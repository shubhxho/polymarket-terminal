# PMT — Polymarket Terminal

A trading terminal for [Polymarket](https://polymarket.com): Bloomberg-grade
density in a calm, modern interface. Keyboard-driven, live order books, a
signal scanner, and basket-arbitrage detection.

Read-only: it quotes, charts and analyses markets. It does not place orders and
never touches a private key.

```bash
npm install
npm run dev      # http://localhost:3000
```

No API keys or environment variables — every upstream endpoint is public.

## Using it

Everything is reachable from the command line at the top. Type a function code
and press Enter, or just type a market name to search.

| Code | Key | Args | Screen |
| --- | --- | --- | --- |
| `MON` | F2 | | Market monitor — live board of the highest-turnover markets |
| `SIG` | F3 | | **Signal scanner** — ranked signals, block flow, basket arbitrage |
| `MOV` | F4 | | Biggest probability gainers and losers over 1H / 24H / 1W |
| `DES` | F5 | `<slug>` | Analytics launchpad: chart, book, tape, holders, signals |
| `SRCH` | F6 | `<query>` | Full-text search across events and markets |
| `WATCH` | F7 | | Your pinned markets, quoted live, with trend sparklines |
| `TAS` | F8 | | Consolidated print tape with a size filter |
| `ALRT` | F9 | | Price alerts evaluated against the live feed |
| `CAT` | F10 | `<sector>` | Browse politics, crypto, sports, economics… |
| `PORT` | | `<0x wallet>` | Any wallet's open positions and P&L |
| `HELP` | F1 | | Command reference and keyboard map |

`DES` also accepts a pasted `polymarket.com/event/…` URL.

**Keys:** `/` focuses the command line · `↑`/`↓` move the grid selection ·
`Enter` opens the selected row · `W` pins it to the watchlist ·
`Ctrl+↑`/`Ctrl+↓` recall commands · `Alt+←`/`Alt+→` go back and forward ·
`F1`–`F10` jump straight to a function.

Light and dark themes both ship; the toggle is in the top-right and the choice
persists.

## Design

The interface follows [ami.dev](https://ami.dev): white ground, hairline
borders, 4–6px radii, a grouped left rail for navigation, and a soft search
pill instead of a command prompt. The whole palette lives in CSS custom
properties behind a `data-theme` attribute, so light and dark are the same
components with different values and no component knows which is active.

Three rules hold the density together:

- **Inter for everything, including the numbers.** Its tabular figures hold a
  price column steady, and a table of Inter digits reads far calmer than the
  same table in monospace. `font-variant-numeric: tabular-nums` is global and
  non-negotiable — a column of prices must not shimmer in width as digits tick.
- **Monospace only for machine identifiers** — the command line, token ids,
  condition hashes, wallet addresses. That is the one place a slashed zero
  matters, and the one place ligatures would actively lie by fusing `->` or
  `!=` into glyphs the underlying string doesn't contain.
- **Green and red are reserved for direction.** Nothing decorative may use
  them, so a tint in a table always means a price and never a theme. Fuchsia
  carries every interactive accent instead.


## Signals

`SIG` runs a scanner over the board every 20 seconds. The engine
(`src/lib/signals.ts`) is pure — no I/O, no React — so it runs server-side for
the scanner and client-side on `DES` over data that screen already has.

Two rules shape it: a signal must be **actionable** (restating a column already
on screen is not a signal), and **silence beats noise** (every detector returns
nothing unless its preconditions hold, and thin markets are excluded before
scoring rather than after).

| Signal | Fires when |
| --- | --- |
| `ARB` | A negative-risk basket is mispriced against the $1 it must settle at |
| `SURGE` | 24h volume far exceeds *that market's own* weekly baseline |
| `MOM` | 24h move still running, with the last hour agreeing |
| `REV` | Last hour fighting the 24h move |
| `WHALE` | Net direction of block prints ≥ $10k, as a share of 24h volume |
| `BOOK` | Resting capital skewed ≥35% to one side, weighted by cost |
| `TAIL` | Price at an extreme but still drawing real turnover |
| `EXPY` | Under 72h to resolution and still genuinely uncertain |
| `WIDE` | Spread is a large fraction of mid — costly to cross |

Each market gets a **heat** (0–100, how much it deserves a look) and a **bias**
(−100..100, net direction, positive = bullish on YES).

### On the arbitrage detector

Exactly one leg of a negative-risk event resolves YES, so its YES legs form a
basket worth exactly $1 at settlement. Sell the basket for more than $1, or buy
it for less, and the difference is locked in.

Three guards keep this honest, and all three matter:

- **Only `negRisk` events.** A sports event carries 20+ *unrelated* legs
  (moneyline, spreads, totals) whose prices sum to nothing meaningful. Summing
  those manufactures a fake 15x "arbitrage" on every single game.
- **Both sides must be genuinely quoted.** An illiquid leg reports `bestAsk`
  near 99¢ with no real offer behind it, which inflates the basket and hides a
  phantom edge. Buy-side opportunities require every leg quoted under 99¢.
- **The thinnest leg caps the trade.** Baskets whose tightest leg has no
  resting liquidity are dropped outright, and the remaining ones display that
  number, because it — not the edge — is what bounds the size.

Edges are shown before fees and slippage, and crossing 20+ legs eats a lot of
both. Treat it as a place to look, not a trade ticket.

## How it works

```
src/
  app/api/*        server-side proxies over the Polymarket APIs
  lib/polymarket   fetching + normalisation (the only place upstream shapes exist)
  hooks/           usePoll (REST), useMarketSocket (CLOB websocket)
  components/      terminal shell, shared grid/book/tape widgets
  components/screens/  one file per function code
```

Three upstream sources, all public:

- **Gamma** (`gamma-api.polymarket.com`) — market and event metadata, search.
  Returns `outcomes` / `outcomePrices` / `clobTokenIds` as JSON-encoded
  *strings*, and omits price-change fields entirely when they are zero.
- **CLOB** (`clob.polymarket.com`) — order books and price history.
- **Data API** (`data-api.polymarket.com`) — trades, holders, positions.

Routes proxy rather than calling upstream from the browser: it keeps CORS out of
the picture, lets Next's data cache absorb repeat polls, and means a screen only
ever handles one response shape (`{ok, data, ts}`).

Quotes arrive over the CLOB websocket (`useMarketSocket`), which fans out book
snapshots and price deltas for every subscribed outcome token. Bursts are
accumulated in refs and published to React once per animation frame, so a hot
market re-renders at most 60 times a second no matter how fast it prints. REST
polling continues underneath as a floor, so a dropped socket degrades to a
5-second refresh rather than a frozen screen.

### Conventions worth knowing before editing

- **Green and red mean direction. Nothing else may use them.** Amber is chrome,
  cyan is labels. A flash of colour in this UI always means a price moved.
- Prices are probabilities (0–1) internally and rendered in cents at the edge.
  Change columns are in *probability points*, not percent.
- Everything is `tabular-nums` and fixed-width; columns must not shimmer as
  digits change.
- `EXPIRED` vs `PENDING`: Polymarket sets `endDate` to the scheduled event time,
  but the book often stays open for hours afterwards awaiting resolution. Past
  end date + still accepting orders renders as PENDING.

## Caveats

- Alerts evaluate on the live socket and therefore only fire while a tab is
  open. They detect *crossings*, so arming a target the market has already
  passed will not fire until it crosses back.
- `MOV` excludes markets pinned within 1.5 points of 0 or 100 — those are
  decided, not moving.
- Watchlist and alerts live in `localStorage`, per browser.
