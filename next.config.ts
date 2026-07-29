import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow this LAN origin to hit the dev server (cross-origin dev requests).
  allowedDevOrigins: ["172.16.0.146"],
  // Linting is owned by oxlint (`npm run lint`), not ESLint. This Next version
  // exposes no `eslint` config key, but with ESLint uninstalled `next build`
  // has nothing to run anyway.
};

export default nextConfig;
