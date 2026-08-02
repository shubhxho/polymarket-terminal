import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow this LAN origin to hit the dev server (cross-origin dev requests).
  allowedDevOrigins: ["172.16.0.146"],
  // Linting is owned by oxlint (`npm run lint`), not ESLint. This Next version
  // exposes no `eslint` config key, but with ESLint uninstalled `next build`
  // has nothing to run anyway.
  experimental: {
    // Tree-shake the animation and data layers to their used surface so a
    // named import doesn't drag the whole barrel into the client bundle.
    optimizePackageImports: ["motion", "@tanstack/react-query"],
    // TypeScript 7 ships the native (tsgo) compiler, which drops the legacy
    // programmatic compiler API `next build` reaches for by default. Run the
    // project-local `tsc` CLI for the build's type-check phase instead.
    useTypeScriptCli: true,
  },
};

export default nextConfig;
