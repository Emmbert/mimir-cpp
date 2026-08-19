#!/usr/bin/env bash
#
# run_benchmarks.sh
#
# Runs the seeded, seeded_parallel, and seeded_distributed latency
# benchmarks one after another against a given parameter file.
#
# Usage:
#   ./run_benchmarks.sh --paramfile ../parameter_files/mimirI_lambda\=230.85_...json
#   ./run_benchmarks.sh --paramfile "../parameter_files/mimirI_lambda=230.85_...json" --threads 32
#   ./run_benchmarks.sh --paramfile "../parameter_files/mimirI_lambda=230.85_...json" --threads 32 --numservers 100

#
# Options:
#   --paramfile PATH   Path to the parameter JSON file (required)
#   --threads N     Number of threads used for the prallel and distributed benchmark (default: 32)
#   --numservers N     Number of servers, passed to the distributed benchmark only
#   -h, --help         Show this help message

set -euo pipefail

THREADS=32
NUMSERVERS="100"
PARAMFILE=""

usage() {
    grep '^#' "$0" | sed -e 's/^#//' -e '1d'
    exit 1
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --paramfile)
            PARAMFILE="$2"
            shift 2
            ;;
        --paramfile=*)
            PARAMFILE="${1#*=}"
            shift
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --threads=*)
            THREADS="${1#*=}"
            shift
            ;;
        --numservers)
            NUMSERVERS="$2"
            shift 2
            ;;
        --numservers=*)
            NUMSERVERS="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ -z "$PARAMFILE" ]]; then
    echo "Error: --paramfile is required." >&2
    usage
fi

if [[ ! -f "$PARAMFILE" ]]; then
    echo "Error: parameter file not found: $PARAMFILE" >&2
    exit 1
fi

echo "Using parameter file: $PARAMFILE"
echo "Threads: $THREADS"
if [[ -n "$NUMSERVERS" ]]; then
    echo "Num servers (distributed only): $NUMSERVERS"
fi
echo

# --- Run benchmarks sequentially ---

echo "==> Running benchmark_latency_seeded ..."
OMP_NUM_THREADS=1 krenew -- nice -n1 ./benchmark_latency_seeded "$PARAMFILE"
echo "==> benchmark_latency_seeded done."
echo

echo "==> Running benchmark_latency_seeded_parallel ..."
OMP_NUM_THREADS="$THREADS" krenew -- nice -n1 ./benchmark_latency_seeded_parallel "$PARAMFILE"
echo "==> benchmark_latency_seeded_parallel done."
echo

echo "==> Running benchmark_latency_seeded_distributed ..."
if [[ -n "$NUMSERVERS" ]]; then
    OMP_NUM_THREADS="$THREADS" krenew -- nice -n1 ./benchmark_latency_seeded_distributed "$PARAMFILE"  "$NUMSERVERS"
else
    OMP_NUM_THREADS="$THREADS" krenew -- nice -n1 ./benchmark_latency_seeded_distributed "$PARAMFILE"
fi
echo "==> benchmark_latency_seeded_distributed done."
echo

echo "All benchmarks completed successfully."