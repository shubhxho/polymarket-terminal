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

**Keys:** `⌘K` or `/` opens the palette · `↑`/`↓` move the grid selection ·
`Enter` opens the selected row · `⌘Enter` opens it in a new tab · `W` pins to
the watchlist · `⌘T`/`⌘W` new and close tab · `⌘1`–`⌘9` jump to a tab ·
`⌥⌘←`/`⌥⌘→` cycle tabs · `⌥←`/`⌥→` go back and forward *within* a tab ·
`F1`–`F10` jump straight to a function.

Light and dark themes both ship; the toggle is in the top-right and the choice
persists.

## Design

The interface follows [ami.dev](https://ami.dev): white ground, hairline
borders, 4–6px radii, a grouped left rail, and a search pill rather than a
prompt. The whole palette lives in CSS custom properties behind a `data-theme`
attribute, so light and dark are the same components with different values.

### Colour is computed, not chosen

The chart palette was **validated with a script, not an eye**. The previous
hand-picked series colours failed hard: worst adjacent pair ΔE 2.7 under
deuteranopia (pink and teal were the same colour to ~5% of men), five of eight
outside the lightness band, all eight under 3:1 contrast.

The replacement is the reference categorical palette, re-ordered for this app.
Orderings were enumerated under one constraint — **green and red must occupy
the last two slots** — and scored by minimum adjacent CVD ΔE in both modes. The
winner scores **13.3 (light) / 23.6 (dark)**, clearing the ≥12 target on both,
where the unconstrained reference ordering only reaches 10.3 in dark.

That constraint is the point: green and red mean *direction* everywhere else in
this terminal, so a chart with a handful of legs must never paint a line in a
hue that reads as "up" beside a table where it would mean exactly that.

### Type

Inter carries the whole interface, **including the numbers** — its tabular
figures hold a price column steady and read far calmer than monospace.
`font-variant-numeric: tabular-nums` is global and non-negotiable.

Monospace is reserved for genuine machine identifiers: token ids, condition
hashes, wallet addresses. That is the one place a slashed zero earns its keep,
and the one place ligatures would actively lie by fusing `->` or `!=` into
glyphs the underlying string does not contain.


## Signals

`SIG` runs a scanner over the board every 20 seconds. The engine is split in
two: `lib/quant.ts` holds the statistics (what is *true*), `lib/signals.ts`
holds the detectors (what is *interesting*). Both are pure — no I/O, no React —
so the scanner runs them server-side and the detail screen runs them in the
browser over data it already fetched.

Three rules shape it:

1. **A signal must be actionable.** "Price went up" is not a signal; "price
   went up on volume in the 97th percentile of today's cross-section, with the
   book still bid" is.
2. **Unusual is measured against a population, not a constant.** Thresholds are
   robust z-scores — median/MAD, not mean/σ, so one market doing 200× normal
   volume can't inflate the scale and hide every other outlier behind it.
3. **Silence beats noise.** Every detector returns nothing unless its
   preconditions hold, and thin markets are excluded before scoring.

| Signal | Fires when |
| --- | --- |
| `ARB` | A negative-risk basket is crossably mispriced against the $1 it settles at |
| `DRIFT` | Basket mids have wandered further from 100¢ than the basket's own quoting noise |
| `SURGE` | Turnover unusual against today's cross-section, not against a fixed multiple |
| `MOM` | Drift **per unit of volatility**, confirmed by positive lag-1 autocorrelation |
| `REV` | Stretched past its Bollinger band while increments mean-revert |
| `BRK` | Stretched *and* still trending, with volatility expanding |
| `COIL` | Volatility compressed below 55% of its earlier level, price still mid-range |
| `WHALE` | Net direction of blocks ≥ $10k, scaled by turnover and one-sidedness |
| `BOOK` | Resting capital skewed ≥35%, weighted by what each side actually risks |
| `LEAN` | Micro-price sits >18% of the way across the spread — next-tick pressure |
| `TAIL` | Extreme price still drawing real turnover |
| `EXPY` | Under 72h to resolution and still genuinely uncertain |
| `THIN` | Spread is a large fraction of mid — costly to cross |

Every signal carries **strength** (how loud) and **confidence** (how much the
inputs justify it — short history, a thin book or three prints all reduce it).
Contributions are scaled by both, so a loud reading built on twelve data points
cannot outrank a quiet one built on six hundred.

Each market then gets three numbers, which answer different questions:

- **heat** (0–100) — does this deserve a look at all?
- **bias** (−100..100) — which way, positive = bullish on YES?
- **conviction** (0–100) — do the directional signals *agree*? Four signals
  pointing the same way is a different proposition from four that cancel, and a
  single composite score cannot tell those apart.

### Why the statistics are the way they are

- **Simple differences, not log returns.** Prices here are probabilities. A
  market moving 2¢→4¢ has doubled in log terms but moved two points of
  probability; treating that as a 69% move would let every longshot dominate
  the volatility ranking.
- **Volatility is scaled by the actual sampling interval**, so a 1h and a 1w
  series are comparable. Without that, "volatility" mostly measures which
  fidelity the caller happened to request.
- **Micro-price weights each side by the *opposite* side's size.** When the bid
  is much larger than the ask, fair value sits nearer the ask, because that is
  where it will trade.
- **Book imbalance is cost-weighted** — a bid risks `price`, an ask risks
  `1 − price` — so a wall of 2¢ asks isn't mistaken for real conviction.

### On the dislocation detectors

Exactly one leg of a negative-risk event resolves YES, so its YES legs form a
basket worth exactly $1 at settlement. `ARB` wants a *crossable* edge on real
quotes; `DRIFT` measures pressure that has built without yet opening one.

Four guards keep both honest, and every one of them was added after the raw
data lied:

- **Only `negRisk` events.** A sports event carries 20+ *unrelated* legs
  (moneyline, spreads, totals) whose prices sum to nothing meaningful. Summing
  those manufactures a fake 15× "arbitrage" on every single game — one Yankees
  event summed to 16.4.
- **Both sides genuinely quoted.** An illiquid leg reports `bestAsk` near 99¢
  with no real offer behind it. Ballon d'Or's 89 legs summed to 59 that way.
- **Every leg two-way for drift.** An unquoted leg's "mid" is 50¢ of fiction —
  on a 128-leg presidential field that produced a basket of 44.46 and a
  4,345-point "drift" before the guard went in.
- **Drift is compared to the basket's own noise.** Each leg's mid could sit
  anywhere within half a spread of true, and those uncertainties accumulate
  with leg count, so a 1-point drift is loud on a 3-leg field and silent on a
  30-leg one. The signal reports the ratio, and fires only above 1×.

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
