#!/usr/bin/env bash
# src/skill/run_tearsheet.sh
# Entry point for tearsheet skill execution

set -euo pipefail

TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 <target>" >&2
  exit 1
fi

python -m src.core.assembler "$TARGET"
