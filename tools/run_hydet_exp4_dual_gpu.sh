#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-all}"
exec "${SCRIPT_DIR}/run_hydet_paper.sh" exp4 hrsc "${ACTION}"
