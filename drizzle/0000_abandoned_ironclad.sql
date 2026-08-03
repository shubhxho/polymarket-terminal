CREATE TABLE "event_log" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"at" timestamp with time zone DEFAULT now() NOT NULL,
	"level" varchar(16) NOT NULL,
	"event" varchar(160) NOT NULL,
	"context" jsonb
);
--> statement-breakpoint
CREATE TABLE "signal_snapshots" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"captured_at" timestamp with time zone DEFAULT now() NOT NULL,
	"scanned" integer NOT NULL,
	"flagged" integer NOT NULL,
	"modeled" integer NOT NULL,
	"confirms" integer NOT NULL,
	"conflicts" integer NOT NULL,
	"bullish" integer NOT NULL,
	"bearish" integer NOT NULL,
	"arbs" integer NOT NULL,
	"drifts" integer NOT NULL,
	"block_notional" double precision NOT NULL,
	"top_market_id" varchar(128),
	"top_heat" integer,
	"markets" jsonb
);
--> statement-breakpoint
CREATE INDEX "event_idx" ON "event_log" USING btree ("event");--> statement-breakpoint
CREATE INDEX "at_idx" ON "event_log" USING btree ("at");--> statement-breakpoint
CREATE INDEX "captured_at_idx" ON "signal_snapshots" USING btree ("captured_at");