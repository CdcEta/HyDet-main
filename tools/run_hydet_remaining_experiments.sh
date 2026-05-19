#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PORT_EXP4="${PORT_EXP4:-29667}"
PORT_EXP2_HRSC="${PORT_EXP2_HRSC:-29677}"
PORT_EXP2_FAIR1M="${PORT_EXP2_FAIR1M:-29687}"
PORT_EXP1_HRSC="${PORT_EXP1_HRSC:-29697}"
PORT_EXP1_FAIR1M="${PORT_EXP1_FAIR1M:-29707}"

PORT="$PORT_EXP4" ./tools/run_hydet_paper.sh exp4 hrsc all
PORT="$PORT_EXP2_HRSC" ./tools/run_hydet_paper.sh exp2 hrsc all
PORT="$PORT_EXP2_FAIR1M" ./tools/run_hydet_paper.sh exp2 fair1m all
PORT="$PORT_EXP1_HRSC" ./tools/run_hydet_paper.sh exp1 hrsc all
PORT="$PORT_EXP1_FAIR1M" ./tools/run_hydet_paper.sh exp1 fair1m all
