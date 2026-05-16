#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate castdet

PORT_EXP4="${PORT_EXP4:-29667}"
PORT_EXP2_HRSC="${PORT_EXP2_HRSC:-29677}"
PORT_EXP2_FAIR1M="${PORT_EXP2_FAIR1M:-29687}"
PORT_EXP1_HRSC="${PORT_EXP1_HRSC:-29697}"
PORT_EXP1_FAIR1M="${PORT_EXP1_FAIR1M:-29707}"

PORT="$PORT_EXP4" ./tools/run_hydet_exp4_dual_gpu.sh all
PORT="$PORT_EXP2_HRSC" ./tools/run_hydet_exp2_dual_gpu.sh hrsc all
PORT="$PORT_EXP2_FAIR1M" ./tools/run_hydet_exp2_dual_gpu.sh fair1m all
PORT="$PORT_EXP1_HRSC" ./tools/run_hydet_exp1_dual_gpu.sh hrsc all
PORT="$PORT_EXP1_FAIR1M" ./tools/run_hydet_exp1_dual_gpu.sh fair1m all
