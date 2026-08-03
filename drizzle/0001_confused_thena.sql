-- Add the per-minute dedup bucket. Non-destructive: add nullable, backfill any
-- existing rows from captured_at, then enforce NOT NULL + the unique index that
-- makes `/api/signals` writes idempotent per minute.
ALTER TABLE "signal_snapshots" ADD COLUMN "minute_bucket" timestamp with time zone;--> statement-breakpoint
UPDATE "signal_snapshots" SET "minute_bucket" = date_trunc('minute', "captured_at") WHERE "minute_bucket" IS NULL;--> statement-breakpoint
ALTER TABLE "signal_snapshots" ALTER COLUMN "minute_bucket" SET NOT NULL;--> statement-breakpoint
CREATE UNIQUE INDEX "snapshot_minute_uq" ON "signal_snapshots" USING btree ("minute_bucket");
