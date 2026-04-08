#!/usr/bin/env bash
# =============================================================================
# run_weekly_sweep.sh — AltBots Weekly Data Refresh Pipeline
#
# Sequences:
#   1. src/utils/url_validator.py   — refresh firms list from fund_managers
#   2. src/scrapers/team_scraper.py — crawl team pages for verified firms
#   3. src/ingestors/team_profiles_ingestor.py — upsert profiles into Qdrant
#
# Error policy: each step is run independently. A failure is logged to
# logs/cron_errors.log but does NOT abort the remaining steps.
#
# Log rotation is applied at the start of each run via a bundled
# logrotate config (config/logrotate.conf). No external cron needed.
#
# A structured heartbeat is written to logs/heartbeat.json at the end
# so the MCP server (check_red_flags / get_tearsheet) can confirm the
# last successful sweep.
#
# Usage:
#   bash run_weekly_sweep.sh          # normal run
#   bash run_weekly_sweep.sh --dry-run  # validate env only, no pipeline steps
#
# Crontab (Sunday 02:00 server time):
#   0 2 * * 0 /root/tearsheet-project/run_weekly_sweep.sh >> /root/tearsheet-project/logs/cron.log 2>&1
# =============================================================================

set -uo pipefail   # treat unset vars as errors, propagate pipe failures
                   # NOTE: -e is intentionally OMITTED so steps can fail without
                   #       aborting the script. Errors are captured manually.

# ── Resolve project root (works whether called directly or via cron) ──────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/venv/bin/python"
LOGS="$PROJECT_DIR/logs"
DATA="$PROJECT_DIR/data"

SWEEP_LOG="$LOGS/weekly_sweep.log"
ERROR_LOG="$LOGS/cron_errors.log"
HEARTBEAT="$LOGS/heartbeat.json"
LOGROTATE_CONF="$PROJECT_DIR/config/logrotate.conf"
LOGROTATE_STATE="$LOGS/.logrotate.state"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Bootstrap ─────────────────────────────────────────────────────────────────
mkdir -p "$LOGS" "$DATA"

# ── Log rotation (run before writing new log content) ────────────────────────
if command -v logrotate &>/dev/null && [[ -f "$LOGROTATE_CONF" ]]; then
    logrotate \
        --state "$LOGROTATE_STATE" \
        "$LOGROTATE_CONF" \
        2>>"$ERROR_LOG" || true   # never abort on rotation failure
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts()  { date '+%Y-%m-%d %H:%M:%S'; }
_log() { echo "[$(_ts)] $*" | tee -a "$SWEEP_LOG"; }
_err() { echo "[$(_ts)] ERROR: $*" | tee -a "$SWEEP_LOG" | tee -a "$ERROR_LOG"; }

# ── Run-state tracking (for heartbeat) ───────────────────────────────────────
RUN_START="$(_ts)"
RUN_START_EPOCH=$(date +%s)
declare -A STEP_STATUS=()
declare -A STEP_DURATION=()

# ── Validate environment ──────────────────────────────────────────────────────
_log "========================================================"
_log "AltBots Weekly Sweep — $RUN_START"
_log "Project:  $PROJECT_DIR"
_log "Venv:     $VENV"
[[ $DRY_RUN -eq 1 ]] && _log "Mode:     DRY-RUN (pipeline steps skipped)"
_log "========================================================"

ENV_OK=1

if [[ ! -x "$VENV" ]]; then
    _err "Python venv not found at $VENV"
    ENV_OK=0
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    _err ".env file missing — API keys will not be loaded"
    # Not fatal; keys may be in the environment already
fi

for required_dir in "$PROJECT_DIR/src" "$DATA"; do
    if [[ ! -d "$required_dir" ]]; then
        _err "Required directory missing: $required_dir"
        ENV_OK=0
    fi
done

if [[ $ENV_OK -eq 0 ]]; then
    _err "Environment validation failed — aborting sweep"
    exit 1
fi

_log "Environment check passed"

if [[ $DRY_RUN -eq 1 ]]; then
    _log "Dry-run complete. Exiting."
    exit 0
fi

# ── Step runner ───────────────────────────────────────────────────────────────
# Usage: run_step <step_name> <command...>
# Sets STEP_STATUS[step_name] to "ok" or "failed:<exit_code>"
run_step() {
    local name="$1"
    shift
    local cmd=("$@")
    local step_start step_end duration exit_code=0

    step_start=$(date +%s)
    _log "-- STEP: $name"
    _log "   Command: ${cmd[*]}"

    # Run inside project dir so relative paths resolve correctly
    (cd "$PROJECT_DIR" && "${cmd[@]}") >> "$SWEEP_LOG" 2>&1 || exit_code=$?

    step_end=$(date +%s)
    duration=$(( step_end - step_start ))
    STEP_DURATION["$name"]="${duration}s"

    if [[ $exit_code -eq 0 ]]; then
        STEP_STATUS["$name"]="ok"
        _log "   ✓ $name completed in ${duration}s"
    else
        STEP_STATUS["$name"]="failed:$exit_code"
        _err "$name failed with exit code $exit_code (duration: ${duration}s)"
        _err "  See $SWEEP_LOG for full output"
    fi
}

# ── Step 1: URL Validator ─────────────────────────────────────────────────────
# Exports verified firm websites from Qdrant fund_managers collection.
# Output: data/firms.json (canonical input for team_scraper)
run_step "url_validator" \
    "$VENV" -m src.utils.url_validator \
    --output "$DATA/firms.json"

# ── Step 2: Team Scraper ──────────────────────────────────────────────────────
# Crawls team pages for firms in data/firms.json via Firecrawl.
# Appends raw markdown records to data/team_profiles_raw.jsonl.
# --resume skips firms already present in the JSONL to avoid duplicate work.
if [[ "${STEP_STATUS[url_validator]}" == "ok" ]] && \
   [[ -f "$DATA/firms.json" ]]; then
    run_step "team_scraper" \
        "$VENV" -m src.scrapers.team_scraper \
        --input "$DATA/firms.json" \
        --resume
else
    _log "-- STEP: team_scraper — SKIPPED (url_validator did not produce firms.json)"
    STEP_STATUS["team_scraper"]="skipped"
    STEP_DURATION["team_scraper"]="0s"
fi

# ── Step 3: Team Profiles Ingestor ────────────────────────────────────────────
# Reads raw JSONL, runs Haiku extraction + bio synthesis, upserts into Qdrant.
# Processes only firms not already in the state file (idempotent).
if [[ -f "$DATA/team_profiles_raw.jsonl" ]]; then
    run_step "team_profiles_ingestor" \
        "$VENV" -m src.ingestors.team_profiles_ingestor \
        --ingest
else
    _log "-- STEP: team_profiles_ingestor — SKIPPED (no team_profiles_raw.jsonl)"
    STEP_STATUS["team_profiles_ingestor"]="skipped"
    STEP_DURATION["team_profiles_ingestor"]="0s"
fi

# ── Tally ─────────────────────────────────────────────────────────────────────
RUN_END="$(_ts)"
RUN_END_EPOCH=$(date +%s)
TOTAL_DURATION=$(( RUN_END_EPOCH - RUN_START_EPOCH ))

FAILED_STEPS=()
for step in url_validator team_scraper team_profiles_ingestor; do
    status="${STEP_STATUS[$step]:-unknown}"
    if [[ "$status" != "ok" && "$status" != "skipped" ]]; then
        FAILED_STEPS+=("$step")
    fi
done

OVERALL_STATUS="success"
[[ ${#FAILED_STEPS[@]} -gt 0 ]] && OVERALL_STATUS="partial_failure"

_log "========================================================"
_log "Sweep complete — $RUN_END"
_log "Total duration: ${TOTAL_DURATION}s"
_log "url_validator:          ${STEP_STATUS[url_validator]:-unknown} (${STEP_DURATION[url_validator]:-?})"
_log "team_scraper:           ${STEP_STATUS[team_scraper]:-unknown} (${STEP_DURATION[team_scraper]:-?})"
_log "team_profiles_ingestor: ${STEP_STATUS[team_profiles_ingestor]:-unknown} (${STEP_DURATION[team_profiles_ingestor]:-?})"
_log "Overall status:         $OVERALL_STATUS"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    _log "Failed steps: ${FAILED_STEPS[*]} — see $ERROR_LOG"
fi
_log "========================================================"

# ── Heartbeat ─────────────────────────────────────────────────────────────────
# Structured JSON written atomically via a temp file.
# Readable by the MCP server, cron monitors, and shell scripts alike.
HEARTBEAT_TMP="${HEARTBEAT}.tmp.$$"
cat > "$HEARTBEAT_TMP" <<EOF
{
  "last_run": "$RUN_END",
  "last_run_epoch": $RUN_END_EPOCH,
  "overall_status": "$OVERALL_STATUS",
  "total_duration_seconds": $TOTAL_DURATION,
  "steps": {
    "url_validator":          { "status": "${STEP_STATUS[url_validator]:-unknown}",          "duration": "${STEP_DURATION[url_validator]:-?}" },
    "team_scraper":           { "status": "${STEP_STATUS[team_scraper]:-unknown}",           "duration": "${STEP_DURATION[team_scraper]:-?}" },
    "team_profiles_ingestor": { "status": "${STEP_STATUS[team_profiles_ingestor]:-unknown}", "duration": "${STEP_DURATION[team_profiles_ingestor]:-?}" }
  },
  "failed_steps": [$(if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then printf '"%s",' "${FAILED_STEPS[@]}" | sed 's/,$//'; fi)]
}
EOF
mv "$HEARTBEAT_TMP" "$HEARTBEAT"
_log "Heartbeat written to $HEARTBEAT"

# ── Final exit code ───────────────────────────────────────────────────────────
# Exit 0 even on partial failure so cron doesn't send duplicate mail.
# Failures are fully captured in cron_errors.log and heartbeat.json.
exit 0
