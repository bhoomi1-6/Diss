import os
import json
import argparse
from datetime import datetime

from tot.tasks import get_task
from tot.methods.dfs import (solve_dfs, solve_dfs_nonparent, solve_dfs_fixed_k2,
                             solve_dfs_nonparent_strict)
from tot.models import claude_usage
from tot.methods.dfs_crossward import (solve_dfs_crossword, solve_dfs_crossword_nonparent,
                                       solve_dfs_crossword_fixed_k2,
                                       solve_dfs_crossword_nonparent_strict, _recurse_crossword,
                                       build_concise_summary)


def run(args):
    task = get_task(args.task, crossword_file=getattr(args, 'crossword_file', None))
    logs, cnt_avg, cnt_any = [], 0, 0

    # Build log filename — timestamped so reruns with identical params
    # never overwrite a previous run, and sorting the folder by name
    # also sorts runs chronologically (most recent last).
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.naive_run:
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_naive_{args.prompt_sample}_sample_{args.n_generate_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_parent_vth{args.v_th}_budget{args.node_budget}'
                f'_eval{args.n_evaluate_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_nonparent':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_diligent_B{args.n_generate_sample}_vth{args.v_th}_budget{args.node_budget}'
                f'_eval{args.n_evaluate_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_fixed_k2':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_fixedk2_B{args.n_generate_sample}_vth{args.v_th}_budget{args.node_budget}'
                f'_eval{args.n_evaluate_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_nonparent_strict':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_strict_B{args.n_generate_sample}_vth{args.v_th}_budget{args.node_budget}'
                f'_eval{args.n_evaluate_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_crossword':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_crossword_prune{not args.no_prune}_maxstate{args.max_per_state}'
                f'_budget{args.node_budget}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_crossword_nonparent':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_crossword_diligent_B{args.n_generate_sample}_prune{not args.no_prune}'
                f'_maxstate{args.max_per_state}_budget{args.node_budget}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_crossword_fixed_k2':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_crossword_fixedk2_B{args.n_generate_sample}_prune{not args.no_prune}'
                f'_maxstate{args.max_per_state}_budget{args.node_budget}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    elif args.method_search == 'dfs_crossword_nonparent_strict':
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_dfs_crossword_strict_B{args.n_generate_sample}_prune{not args.no_prune}'
                f'_maxstate{args.max_per_state}_budget{args.node_budget}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    else:
        file = (f'./logs/{args.task}/{args.backend}_{args.temperature}'
                f'_BFS_{args.method_generate}{args.n_generate_sample}'
                f'_{args.method_evaluate}{args.n_evaluate_sample}'
                f'_{args.method_select}{args.n_select_sample}'
                f'_start{args.task_start_index}_end{args.task_end_index}'
                f'_{timestamp}.json')
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # Concise pandas-friendly per-puzzle summary, crossword conditions only —
    # additive-only: the detailed `logs` file above is unchanged, this is a
    # separate sibling file with a `_summary` suffix so it never overwrites
    # or interferes with the existing detailed log.
    is_crossword_run = args.method_search.startswith('dfs_crossword')
    summary_file = file.replace('.json', '_summary.json') if is_crossword_run else None
    summaries = []

    for i in range(args.task_start_index, args.task_end_index):
        # Solve
        usage_before = claude_usage(args.backend)
  
        if args.method_search == 'dfs':
            ys, info = solve_dfs(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_nonparent':
            ys, info = solve_dfs_nonparent(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_fixed_k2':
            ys, info = solve_dfs_fixed_k2(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_nonparent_strict':
            ys, info = solve_dfs_nonparent_strict(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_crossword':
            ys, info = solve_dfs_crossword(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_crossword_nonparent':
            ys, info = solve_dfs_crossword_nonparent(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_crossword_fixed_k2':
            ys, info = solve_dfs_crossword_fixed_k2(args, task, i, to_print=args.verbose)
        elif args.method_search == 'dfs_crossword_nonparent_strict':
            ys, info = solve_dfs_crossword_nonparent_strict(args, task, i, to_print=args.verbose)
        else:
            raise ValueError(
                f"Unsupported --method_search {args.method_search!r}. "
                "Only 'dfs', 'dfs_nonparent', 'dfs_fixed_k2', 'dfs_nonparent_strict', "
                "'dfs_crossword', 'dfs_crossword_nonparent', 'dfs_crossword_fixed_k2', "
                "and 'dfs_crossword_nonparent_strict' are implemented "
                "(there is no BFS solver in this codebase)."
            )
        usage_after = claude_usage(args.backend)
        usage_this_puzzle = {
            'completion_tokens': usage_after['completion_tokens'] - usage_before['completion_tokens'],
            'prompt_tokens':     usage_after['prompt_tokens']     - usage_before['prompt_tokens'],
            'cost':              usage_after['cost']              - usage_before['cost'],
        }

        # Log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos,
                     'usage_this_puzzle': usage_this_puzzle, 'usage_so_far': usage_after,
                     'config': vars(args)})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)

        if is_crossword_run:
            summaries.append(build_concise_summary(info, args, i, usage_this_puzzle))
            with open(summary_file, 'w') as f:
                json.dump(summaries, f, indent=2)

        # Print metrics
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        nodes = info.get('nodes_explored', '—')
        cand_eval = info.get('candidates_evaluated', '—')
        cand_pruned = info.get('candidates_pruned', '—')
        backtracks = len(info.get('backtracks', []))
        mean_jump = info.get('mean_jump_size', '—')
        pct_gt1 = info.get('pct_jumps_gt1', '—')
        print(i, 'sum(accs)', sum(accs),
              'cnt_avg', cnt_avg,
              'cnt_any', cnt_any,
              f'nodes={nodes}',
              f'candidates_evaluated={cand_eval}',
              f'candidates_pruned={cand_pruned}',
              f'backtracks={backtracks}',
              f'mean_jump={mean_jump}',
              f'pct_jumps_gt1={pct_gt1}',
              f"tokens_this_puzzle={usage_this_puzzle['completion_tokens'] + usage_this_puzzle['prompt_tokens']}",
              f"cost_this_puzzle={usage_this_puzzle['cost']:.5f}", '\n')

    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', claude_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='bedrock')
    args.add_argument('--temperature', type=float, default=0)

    args.add_argument('--task', type=str, required=True,
                      choices=['game24', 'text', 'crosswords'])

    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])


    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select',   type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1) #change to 3 for Game 24, 5 for crosswards
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample',   type=int, default=1)

    # DFS args
    args.add_argument('--method_search', type=str,
                      choices=['bfs', 'dfs', 'dfs_nonparent', 'dfs_fixed_k2', 'dfs_nonparent_strict',
                               'dfs_crossword', 'dfs_crossword_nonparent', 'dfs_crossword_fixed_k2',
                               'dfs_crossword_nonparent_strict'],
                      default='dfs',
                      help='Search algorithm: bfs | dfs (parent-only) | dfs_nonparent (beta(c)) | '
                           'dfs_fixed_k2 (deterministic fixed jump=2, no LLM backtrack call) | '
                           'dfs_nonparent_strict (beta(c), parent forbidden as LLM target; falls back '
                           'to root if no non-parent ancestor exists, or to parent if the model returns '
                           'UNRECOVERABLE/invalid/parent — never aborts the search) | '
                           'dfs_crossword (parent-only) | dfs_crossword_nonparent (beta(c)) | '
                           'dfs_crossword_fixed_k2 (deterministic fixed jump=2, crossword) | '
                           'dfs_crossword_nonparent_strict (beta(c), parent forbidden, crossword, '
                           'same root/parent fallback as dfs_nonparent_strict)')
    args.add_argument('--v_th', type=float, default=0.5,
                      help='Value threshold for DFS pruning (prune if score <= v_th)')
    args.add_argument('--node_budget', type=int, default=30,
                      help='Max nodes to explore per problem in DFS')
    args.add_argument('--verbose', action='store_true', default=True,
                      help='Print step-by-step DFS trace (off by default for large runs)')

    # dfs_crossword args
    args.add_argument('--max_per_state', type=int, default=3,
                      help='Max candidate words to branch into per crossword board state')
    args.add_argument('--no_prune', action='store_true',
                      help='Disable impossible-word pruning for dfs_crossword')
    args.add_argument('--crossword_file', type=str,
                      choices=['mini0505.json', 'mini0505_0_100_5.json'],
                      default='mini0505.json',
                      help="Crossword dataset file (tot/data/crosswords/). "
                           "'mini0505.json' = full 156-puzzle set (current default, unchanged). "
                           "'mini0505_0_100_5.json' = the 20-puzzle held-out subset (indices "
                           "0,5,...,95 of the full set) matching the original ToT paper's "
                           "MiniCrosswords evaluation set. Ignored by non-crossword tasks. "
                           "When using this file, pass --task_start_index 0 --task_end_index 20.")
    args.add_argument('--selection_method', type=str, choices=['deepest', 'best_r_word'],
                      default='deepest',
                      help="Final-state selection for dfs_crossword* methods: 'deepest' "
                           "(main A/B/C/D experiment; deepest explored state, first on tie) "
                           "or 'best_r_word' (original ToT paper's '+best state' oracle "
                           "ablation; NOT used by the main experiment). Ignored by non-crossword "
                           "methods.")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
