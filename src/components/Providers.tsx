"use client";

import { MotionConfig } from "motion/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

/**
 * App-wide client providers, mounted once at the root.
 *
 * - `QueryClientProvider` backs the whole polling layer (`usePoll`). Because
 *   the client is created inside a `useState` initialiser it is stable for the
 *   life of the tab but never shared across requests — the pattern React Query
 *   documents for the App Router, where a module-level singleton would leak one
 *   user's cache into the next on the server.
 * - `MotionConfig reducedMotion="user"` makes every Motion animation in the
 *   tree honour the OS "reduce motion" setting without a single call site
 *   restating the media query.
 *
 * Defaults are tuned for a terminal that keeps stale data on screen: a short
 * `staleTime` (screens re-poll on their own cadence via `refetchInterval`), no
 * retry storm, and `refetchOnWindowFocus` so a returning trader never reads a
 * minutes-old quote.
 */
export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5_000,
            gcTime: 5 * 60_000,
            retry: 1,
            refetchOnWindowFocus: true,
            refetchOnReconnect: true,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={client}>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </QueryClientProvider>
  );
}
