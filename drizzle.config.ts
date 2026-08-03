import "dotenv/config";
import { defineConfig } from "drizzle-kit";

/**
 * Drizzle Kit config — used only by the `db:*` scripts (generate/migrate/push/
 * studio), never at app runtime. Reads DATABASE_URL from `.env` via dotenv.
 */
export default defineConfig({
  out: "./drizzle",
  schema: "./src/db/schema.ts",
  dialect: "postgresql",
  dbCredentials: { url: process.env.DATABASE_URL ?? "" },
});
