#!/usr/bin/env bash
set -euo pipefail

run_bench() {
    local title="$1"
    local script="$2"

    printf "\n========== %s ==========" "$title"
    printf "\n"
    PYTHONPATH=src uv run "$script"
}

run_bench "Elementwise Benchmarks" "tests/bench_elementwise.py"
run_bench "Matmul Benchmarks" "tests/bench_matmul.py"
run_bench "Reduction Benchmarks" "tests/bench_reduction.py"
run_bench "Init Benchmarks" "tests/bench_init.py"
