# Tree of Thoughts: Backtracking Mechanisms in DFS Search (Dissertation Fork)

This repository is a dissertation project built on top of the official
[Tree of Thoughts (ToT)](https://github.com/princeton-nlp/tree-of-thought-llm)
codebase. It studies how an LLM-based DFS search backtracks when it fails
down a branch: specifically, whether letting the model choose *how far* to
jump back up the tree (rather than always retreating one level to the
parent) changes search efficiency and solution quality.

The original ToT paper's BFS/naive-sampling code, tasks, and prompts are
unchanged and still run as documented below. The additions specific to
this dissertation are the four DFS backtracking conditions (A/B/C/D)
implemented in [`src/tot/methods/dfs.py`](src/tot/methods/dfs.py) (Game
of 24) and [`src/tot/methods/dfs_crossward.py`](src/tot/methods/dfs_crossward.py)
(Mini Crosswords), the AWS Bedrock backend in
[`src/tot/models.py`](src/tot/models.py), and the experiment-management
scripts and logs described below.

> **Note on live reproduction:** the AWS Bedrock inference profile used for
> this dissertation's runs has since been deleted, and the associated AWS
> credits are exhausted. The setup and commands below are correct and were
> used to produce every result in `logs/final_experiment/`, but re-running
> them requires your own AWS account with Bedrock model access and a new
> inference profile/model ARN (see Setup, step 2-3) — the code itself is
> not tied to my account. The complete raw logs and summaries for all four
> conditions on both benchmarks are committed under `logs/final_experiment/`
> so the results can be verified and re-analyzed without needing to re-run
> any LLM calls.

## The four backtracking conditions

Both benchmarks implement the same four conditions via `--method_search`:

| Condition | `--method_search` (game24 / crosswords) | Behaviour |
|---|---|---|
| **A — Parent** | `dfs` / `dfs_crossword` | Always backtracks exactly one level, to the immediate parent. No model call for backtracking. |
| **B — β(c), unconstrained** | `dfs_nonparent` / `dfs_crossword_nonparent` | Model is asked to choose which ancestor to jump back to; the parent is an allowed answer. |
| **C — Fixed k=2** | `dfs_fixed_k2` / `dfs_crossword_fixed_k2` | Deterministically jumps back exactly 2 levels. **No model call at all** — this is the only condition that is model-free. |
| **D — β(c), constrained** | `dfs_nonparent_strict` / `dfs_crossword_nonparent_strict` | Model chooses an ancestor as in B, but the parent is rejected as an illegal answer; falls back deterministically (to the nearest legal ancestor, or the root) only when the model returns an invalid/unparseable/parent response. Still a model call every step — **this condition is model-guided, just constrained**, not "non-model." |

A and C never call the model to decide where to backtrack; B and D do.
The axis that actually separates C and D from B in the results is
*whether a parent-equivalent jump can ever occur* (C and D structurally
avoid it; B's prompt allows it and the model uses it often) — not a
model/non-model split, since D is model-guided.

## Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone <this-repo-url>
   cd tree-of-thought-llm
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .   # installs the `tot` package
   ```

2. **Configure AWS Bedrock access.** This fork calls Claude through AWS
   Bedrock rather than the OpenAI API. You need:
   - AWS credentials with `bedrock:InvokeModel` permission, available via
     the normal AWS credential chain (e.g. `aws configure`, an
     `AWS_PROFILE`, or environment variables `AWS_ACCESS_KEY_ID` /
     `AWS_SECRET_ACCESS_KEY`).
   - A Bedrock model ID or application inference profile ARN for a Claude
     model, in a region where you have model access enabled.

3. **Set the model ARN.** Copy the example env file and fill in your own
   values — do not hardcode this in source, it identifies your AWS
   account:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   BEDROCK_MODEL_ARN=arn:aws:bedrock:<region>:<your-account-id>:application-inference-profile/<your-profile-id>
   BEDROCK_REGION=us-east-1
   ```
   `.env` is gitignored and loaded automatically (`src/tot/models.py`
   calls `load_dotenv()`).

4. **Smoke-test the setup** on 2 puzzles before running anything large:
   ```bash
   ./scripts/run_small_demo.sh game24
   ./scripts/run_small_demo.sh crosswords
   ```

## Reproducing the dissertation experiments

Each condition is run via `run.py` with `--task {game24,crosswords}` and
the corresponding `--method_search` value from the table above. The
configuration actually used for the final experiment (puzzle ranges,
sample counts, node budget, pruning, etc.) is recorded in
[`logs/final_experiment/EXPERIMENT_MANIFEST.json`](logs/final_experiment/EXPERIMENT_MANIFEST.json),
including the exact CLI command template for each benchmark. In summary:

```bash
# Game of 24, puzzles 900-999, all four conditions:
python run.py --task game24 --method_search dfs                    --task_start_index 900 --task_end_index 1000 --n_generate_sample 3 --node_budget 50 --backend bedrock --verbose
python run.py --task game24 --method_search dfs_nonparent          --task_start_index 900 --task_end_index 1000 --n_generate_sample 3 --node_budget 50 --backend bedrock --verbose
python run.py --task game24 --method_search dfs_fixed_k2           --task_start_index 900 --task_end_index 1000 --n_generate_sample 3 --node_budget 50 --backend bedrock --verbose
python run.py --task game24 --method_search dfs_nonparent_strict   --task_start_index 900 --task_end_index 1000 --n_generate_sample 3 --node_budget 50 --backend bedrock --verbose

# Mini Crosswords, 20-puzzle held-out set, all four conditions:
python run.py --task crosswords --method_search dfs_crossword                  --crossword_file mini0505_0_100_5.json --task_start_index 0 --task_end_index 20 --n_generate_sample 5 --node_budget 50 --max_per_state 3 --backend bedrock --verbose
python run.py --task crosswords --method_search dfs_crossword_nonparent        --crossword_file mini0505_0_100_5.json --task_start_index 0 --task_end_index 20 --n_generate_sample 5 --node_budget 50 --max_per_state 3 --backend bedrock --verbose
python run.py --task crosswords --method_search dfs_crossword_fixed_k2         --crossword_file mini0505_0_100_5.json --task_start_index 0 --task_end_index 20 --n_generate_sample 5 --node_budget 50 --max_per_state 3 --backend bedrock --verbose
python run.py --task crosswords --method_search dfs_crossword_nonparent_strict --crossword_file mini0505_0_100_5.json --task_start_index 0 --task_end_index 20 --n_generate_sample 5 --node_budget 50 --max_per_state 3 --backend bedrock --verbose
```

`v_th` (pruning threshold, 0.5) and `n_evaluate_sample` (1) are left at
their `run.py` defaults for both benchmarks, per the manifest. Add
`--resume` to any command to skip puzzle indices already recorded in that
run's checkpoint file, if a run was interrupted.

### Recording a run under the final-experiment naming convention

Rather than calling `run.py` directly for a run you want to keep,
use the wrapper, which copies the new log files it produced into
`logs/final_experiment/<task>/<condition>/` under the standardized
filename described in
[`logs/final_experiment/NAMING.md`](logs/final_experiment/NAMING.md)
(your originals in `logs/<task>/` are left untouched):

```bash
python scripts/run_final_experiment.py --task game24 --method_search dfs_nonparent \
    --task_start_index 900 --task_end_index 1000 --n_generate_sample 3 \
    --node_budget 50 --backend bedrock --verbose
```

## Logs and results

- `logs/<task>/` — raw output of every `run.py` invocation (pilot runs,
  dev runs, and final runs alike), named by `run.py` itself.
- `logs/final_experiment/<task>/<condition>/` — the curated final-run
  logs for each of the four conditions, one detail JSON + one summary
  JSON per condition (crosswords also gets a concise per-puzzle
  `_summary.json`; see `NAMING.md` for the exact convention).
- `logs/final_experiment/EXPERIMENT_MANIFEST.json` — the intended
  configuration matrix for the final experiment (puzzle ranges, sample
  counts, node budget, pruning, selection method) with the reasoning
  behind each confirmed choice.

## Statistical analysis

The notebooks in [`statistical_analysis/`](statistical_analysis/) consume
the logs under `logs/final_experiment/` to produce the per-benchmark and
cross-benchmark results used in the dissertation:
- `Statistical_analysis_game24.ipynb`
- `Statistical_analysis_crosswards.ipynb`
- `Statistical_analysis_cross_benchmark.ipynb`

## Adding a new task

Unchanged from upstream ToT — see the original instructions:
* Add a task class in `tot/tasks/` and its data files in `tot/data/`
  (see `tot/tasks/game24.py`), and register it in `tot/tasks/__init__.py`.
* Add task-specific prompts in `tot/prompts/` (see `tot/prompts/game24.py`).

## Citation

This work builds directly on the Tree of Thoughts paper and codebase.
If you use this repository, please cite the original paper:

```bibtex
@misc{yao2023treethoughtsdeliberateproblem,
      title={Tree of Thoughts: Deliberate Problem Solving with Large Language Models}, 
      author={Shunyu Yao and Dian Yu and Jeffrey Zhao and Izhak Shafran and Thomas L. Griffiths and Yuan Cao and Karthik Narasimhan},
      year={2023},
      eprint={2305.10601},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.10601}, 
}
```

Paper: [arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
Original codebase: [github.com/princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm)
