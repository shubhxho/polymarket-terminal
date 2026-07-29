import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Linting is owned by oxlint (`npm run lint`), not ESLint. This Next version
  // exposes no `eslint` config key, but with ESLint uninstalled `next build`
  // has nothing to run anyway.
};

export default nextConfig;
