#!/usr/bin/env python3
"""
Wrapper around run.py for FINAL-EXPERIMENT runs only.

Does NOT modify run.py, any search logic, or the logging internals inside
run.py -- it calls run.py exactly as-is, via subprocess, with whatever
arguments you pass through. Pilot/dev runs are completely unaffected by
this script's existence; keep calling `python run.py ...` directly for
those, exactly as before.

What this script adds, purely as external housekeeping:
  1. Records the file listing in logs/<task>/ before the run.
  2. Invokes `python run.py <args...>` (all args forwarded verbatim).
  3. Diffs the directory afterward to find the newly-created log file(s)
     (the detailed log, and for crossword conditions the sibling
     `_summary.json` -- both are picked up automatically by the diff).
  4. COPIES (never moves) them into
     logs/final_experiment/<task>/<condition_dir>/, renamed to the
     FINAL-EXPERIMENT naming convention documented in
     logs/final_experiment/NAMING.md. The original file in logs/<task>/
     is left untouched.

Usage: identical to `python run.py ...` -- just invoke this script instead.

    python scripts/run_final_experiment.py --task game24 \\
        --method_search dfs --task_start_index 900 --task_end_index 1000 \\
        --node_budget 30 --backend bedrock
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

METHOD_TO_CONDITION_DIR = {
    'dfs':                              'A_parent',
    'dfs_crossword':                    'A_parent',
    'dfs_nonparent':                    'B_beta_c',
    'dfs_crossword_nonparent':          'B_beta_c',
    'dfs_fixed_k2':                     'C_fixed_k2',
    'dfs_crossword_fixed_k2':           'C_fixed_k2',
    'dfs_nonparent_strict':             'D_strict_nonparent',
    'dfs_crossword_nonparent_strict':   'D_strict_nonparent',
}

_TIMESTAMP_PATTERN = re.compile(r'(\d{8}_\d{6})')


def parse_task_and_method(argv):
    task = method = None
    for i, a in enumerate(argv):
        if a == '--task' and i + 1 < len(argv):
            task = argv[i + 1]
        if a == '--method_search' and i + 1 < len(argv):
            method = argv[i + 1]
    return task, method


def _load_config(detail_json_path):
    """
    Read the actual `config` dict (vars(args), written by run.py on every
    puzzle) out of the detail log itself -- the authoritative source of
    what values were really used for this run. Never regex-parses the
    filename for these values: some conditions (e.g. Condition A) don't
    embed every field in their filename at all (n_generate_sample is
    absent from Condition A's template), so filename-parsing would have to
    guess or silently default -- this reads the true value instead.
    """
    with open(detail_json_path) as f:
        data = json.load(f)
    if not data:
        return {}
    return data[0].get('config', {})


def rename_to_convention(detail_json_path, cond_dir):
    """
    Build the FINAL-EXPERIMENT filename for the detail log at
    detail_json_path, using ONLY values read from its own embedded
    `config` dict (see _load_config) plus the puzzle range/timestamp
    (also authoritative: task_start_index/task_end_index are in config
    too; the timestamp is pulled from run.py's own filename since it
    isn't otherwise recorded in the JSON body).
    """
    original_name = os.path.basename(detail_json_path)
    cfg = _load_config(detail_json_path)
    task = cfg.get('task', 'unknowntask')

    ts_match = _TIMESTAMP_PATTERN.search(original_name)
    ts = ts_match.group(1) if ts_match else 'NOTIMESTAMP'

    parts = [
        task, cond_dir,
        str(cfg.get('backend', 'unknownbackend')),
        f"B{cfg.get('n_generate_sample', 'NA')}",
        f"E{cfg.get('n_evaluate_sample', 'NA')}",
        f"budget{cfg.get('node_budget', 'NA')}",
        f"vth{cfg.get('v_th', 'NA')}",
    ]
    if task == 'crosswords':
        # max_per_state / no_prune are always present in cfg (global argparse
        # args) but are only semantically meaningful for crossword runs --
        # Game24 never reads them, so omit them from Game24 filenames rather
        # than implying a relevance they don't have.
        if 'max_per_state' in cfg:
            parts.append(f"maxstate{cfg['max_per_state']}")
        if 'no_prune' in cfg:
            parts.append(f"prune{not cfg['no_prune']}")
    parts += [str(cfg.get('task_start_index', 'NA')), str(cfg.get('task_end_index', 'NA')), ts]
    return '_'.join(parts) + '.json'


def main():
    argv = sys.argv[1:]
    task, method = parse_task_and_method(argv)
    if not task or not method:
        print("ERROR: --task and --method_search are required", file=sys.stderr)
        sys.exit(1)
    if method not in METHOD_TO_CONDITION_DIR:
        print(f"ERROR: unrecognized --method_search {method!r}", file=sys.stderr)
        sys.exit(1)

    cond_dir = METHOD_TO_CONDITION_DIR[method]
    src_dir = os.path.join('logs', task)
    dest_dir = os.path.join('logs', 'final_experiment', task, cond_dir)
    os.makedirs(dest_dir, exist_ok=True)

    before = set(glob.glob(os.path.join(src_dir, '*.json')))

    result = subprocess.run([sys.executable, 'run.py'] + argv)
    if result.returncode != 0:
        print("run.py exited non-zero -- not copying anything.", file=sys.stderr)
        sys.exit(result.returncode)

    after = set(glob.glob(os.path.join(src_dir, '*.json')))
    new_files = sorted(after - before)
    if not new_files:
        print("WARNING: no new files detected in logs/%s/ -- nothing copied." % task,
              file=sys.stderr)
        sys.exit(1)

    detail_files = [f for f in new_files if not f.endswith('_summary.json')]
    summary_files = [f for f in new_files if f.endswith('_summary.json')]

    print(f"\nCopying {len(new_files)} new file(s) to {dest_dir}/ "
          f"(originals in {src_dir}/ left untouched):")

    renamed_by_stem = {}  # original detail-file basename (no ext) -> new base name (no ext)
    for f in detail_files:
        renamed = rename_to_convention(f, cond_dir)
        dest_path = os.path.join(dest_dir, renamed)
        shutil.copy2(f, dest_path)
        print(f"  {os.path.basename(f)}  ->  {dest_path}")
        stem = os.path.basename(f)[:-len('.json')]
        renamed_by_stem[stem] = renamed[:-len('.json')]

    for f in summary_files:
        base = os.path.basename(f)
        detail_stem = base[:-len('_summary.json')]
        if detail_stem in renamed_by_stem:
            renamed = renamed_by_stem[detail_stem] + '_summary.json'
        else:
            # Paired detail file wasn't among the new files (unexpected) --
            # fall back to reading this run's own config indirectly isn't
            # possible (summaries don't carry it), so keep the original
            # name rather than guess.
            renamed = base
        dest_path = os.path.join(dest_dir, renamed)
        shutil.copy2(f, dest_path)
        print(f"  {base}  ->  {dest_path}")


if __name__ == '__main__':
    main()
