import "dotenv/config";
import { defineConfig } from "drizzle-kit";
import { pgConnectionUrl } from "./src/db/url";

/**
 * Drizzle Kit config — used only by the `db:*` scripts (generate/migrate/push/
 * studio), never at app runtime. Reads DATABASE_URL from `.env` via dotenv.
 */
export default defineConfig({
  out: "./drizzle",
  schema: "./src/db/schema.ts",
  dialect: "postgresql",
  dbCredentials: { url: pgConnectionUrl(process.env.DATABASE_URL ?? "") },
});
