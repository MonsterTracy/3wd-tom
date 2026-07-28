#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash run_batch.sh <config_path> <log_root_dir> <num_games> [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  bash run_batch.sh configs/deepseek_vs_twdm.yaml logs/deepseek_vs_twdm 10"
    echo "  bash run_batch.sh configs/gpt_vs_twdm.yaml logs/gpt_vs_twdm 10"
    exit 1
fi

game_config="$1"
game_dir="$2"
num="$3"
shift 3

mkdir -p "$game_dir"

for ((i=1; i<=num; i++))
do
    run_dir="$game_dir/game_${i}"
    mkdir -p "$run_dir"

    echo "Running game ${i}/${num}"
    echo "Config: $game_config"
    echo "Log dir: $run_dir"

    python run_battle.py \
        --config "$game_config" \
        --log_save_path "$run_dir" \
        "$@"
done
