# Polymarket Terminal

A Bloomberg-style terminal for [Polymarket](https://polymarket.com) prediction markets, built with Next.js 16 (App Router) and Tailwind CSS 4. No API keys required — it reads Polymarket's public APIs.

## Features

- **Dashboard** (`/`) — top markets by 24h volume with leader outcome, implied odds, 24h move, volume, liquidity and end date. Aggregate stats across the board.
- **Category filters** — politics, sports, crypto, economy, geopolitics, tech, culture (via Gamma `tag_slug`).
- **Search** — full-text market search via Gamma `public-search`.
- **Event pages** (`/event/[slug]`) — server-rendered SVG price-history chart (1D / 1W / 1M / MAX, top 5 outcomes for multi-outcome events), outcome table with bid/ask/spread, resolution rules, and a link out to trade.

## Data sources

- `gamma-api.polymarket.com` — events, markets, prices, tags (revalidated every 30–60s)
- `clob.polymarket.com` — price history time series (revalidated every 120s)

## Development

```bash
bun install
bun dev        # http://localhost:3000
bun run lint   # biome check
bun run build  # production build
```
