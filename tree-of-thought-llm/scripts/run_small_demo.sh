set -e
cd "$(dirname "$0")/.."

benchmark="$1"

if [ "$benchmark" = "game24" ]; then
    python run.py \
        --task game24 \
        --method_search dfs_nonparent \
        --task_start_index 900 \
        --task_end_index 902 \
        --n_generate_sample 3 \
        --node_budget 20 \
        --backend bedrock \
        --verbose

elif [ "$benchmark" = "crosswords" ]; then
    python run.py \
        --task crosswords \
        --method_search dfs_crossword_nonparent \
        --crossword_file mini0505_0_100_5.json \
        --task_start_index 0 \
        --task_end_index 2 \
        --n_generate_sample 5 \
        --node_budget 20 \
        --max_per_state 3 \
        --backend bedrock \
        --verbose

else
    echo "Usage: $0 {game24|crosswords}" >&2
    exit 1
fi
