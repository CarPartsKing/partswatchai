-- =============================================================================
-- Migration 003: data_quality_issues table
-- Run this in the Supabase SQL Editor before using transform/clean.py
-- Safe to re-run — all statements are idempotent.
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id            BIGSERIAL    PRIMARY KEY,
    checked_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Which record has the problem
    source_table  TEXT         NOT NULL,     -- e.g. 'sales_transactions'
    source_id     TEXT         NOT NULL,     -- e.g. transaction_id value

    -- What the problem is
    issue_type    TEXT         NOT NULL,     -- machine-readable tag (stable key)
    issue_detail  TEXT,                      -- human-readable description
    field_name    TEXT,                      -- which column has the bad value
    field_value   TEXT,                      -- the bad value as a string

    -- Triage
    severity      TEXT         NOT NULL DEFAULT 'warning'
                               CHECK (severity IN ('warning', 'error')),
    is_resolved   BOOLEAN      NOT NULL DEFAULT FALSE,
    resolved_at   TIMESTAMPTZ,
    resolved_by   TEXT,

    -- Upsert key: same record + same rule = one row, updated on each run
    UNIQUE (source_table, source_id, issue_type)
);

-- Indexes to support common queries:
--   "show me all unresolved errors"
--   "show me all issues of type X"
--   "show me everything flagged in the last run"
CREATE INDEX IF NOT EXISTS idx_dqi_source_table  ON data_quality_issues (source_table);
CREATE INDEX IF NOT EXISTS idx_dqi_issue_type    ON data_quality_issues (issue_type);
CREATE INDEX IF NOT EXISTS idx_dqi_severity      ON data_quality_issues (severity);
CREATE INDEX IF NOT EXISTS idx_dqi_is_resolved   ON data_quality_issues (is_resolved);
CREATE INDEX IF NOT EXISTS idx_dqi_checked_at    ON data_quality_issues (checked_at DESC);

-- Disable RLS (consistent with all other tables in this project)
ALTER TABLE data_quality_issues DISABLE ROW LEVEL SECURITY;
