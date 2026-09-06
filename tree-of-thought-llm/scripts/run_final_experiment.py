#!/usr/bin/env python3
import glob
import json
import os
import re
import shutil
import subprocess
import sys


# Map each method name to the condition folder used in the final results.
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


# Used to keep the timestamp from the original log filename.
_TIMESTAMP_PATTERN = re.compile(r'(\d{8}_\d{6})')


def parse_task_and_method(argv):

    #pick out the task and the method from the command line arguments
    task = method = None
    for i, a in enumerate(argv):
        if a == '--task' and i + 1 < len(argv):
            task = argv[i + 1]
        if a == '--method_search' and i + 1 < len(argv):
            method = argv[i + 1]
    return task, method


def _load_config(detail_json_path):
    """
    Read the config saved inside the detail log.

    This is the actual configuration used for the run, so it is safer
    than trying to work out the settings from the filename.
    """
    with open(detail_json_path) as f:
        data = json.load(f)
    if not data:
        return {}
    return data[0].get('config', {})


def rename_to_convention(detail_json_path, cond_dir):
    """
    Create the final filename for a detail log using the settings
    stored in that log, together with the original run timestamp.
    """
    original_name = os.path.basename(detail_json_path)
    cfg = _load_config(detail_json_path)
    task = cfg.get('task', 'unknowntask')

    ts_match = _TIMESTAMP_PATTERN.search(original_name)
    ts = ts_match.group(1) if ts_match else 'NOTIMESTAMP'


     # Build the filename from the settings that were actually used.
    parts = [
        task, cond_dir,
        str(cfg.get('backend', 'unknownbackend')),
        f"B{cfg.get('n_generate_sample', 'NA')}",
        f"E{cfg.get('n_evaluate_sample', 'NA')}",
        f"budget{cfg.get('node_budget', 'NA')}",
        f"vth{cfg.get('v_th', 'NA')}",
    ]
    if task == 'crosswords':
        # These settings only matter for the crossword experiments,
        # so they are left out of the Game of 24 filenames.
        if 'max_per_state' in cfg:
            parts.append(f"maxstate{cfg['max_per_state']}")
        if 'no_prune' in cfg:
            parts.append(f"prune{not cfg['no_prune']}")
    parts += [str(cfg.get('task_start_index', 'NA')), str(cfg.get('task_end_index', 'NA')), ts]
    return '_'.join(parts) + '.json'


def main():
    argv = sys.argv[1:]

    # Get the task and method from the command-line arguments.
    task, method = parse_task_and_method(argv)

    if not task or not method:
        print("ERROR: --task and --method_search are required", file=sys.stderr)
        sys.exit(1)

    if method not in METHOD_TO_CONDITION_DIR:
        print(f"ERROR: unrecognized --method_search {method!r}", file=sys.stderr)
        sys.exit(1)

    # Work out where the original logs are and where the final copies go.
    cond_dir = METHOD_TO_CONDITION_DIR[method]
    src_dir = os.path.join('logs', task)
    dest_dir = os.path.join('logs', 'final_experiment', task, cond_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Remember which files already exist before starting the run.
    before = set(glob.glob(os.path.join(src_dir, '*.json')))

    result = subprocess.run([sys.executable, 'run.py'] + argv)
    if result.returncode != 0:
        print("run.py exited non-zero -- not copying anything.", file=sys.stderr)
        sys.exit(result.returncode)

     # Anything that appeared after the run is a new log file.
    after = set(glob.glob(os.path.join(src_dir, '*.json')))
    new_files = sorted(after - before)
    if not new_files:
        print("WARNING: no new files detected in logs/%s/ -- nothing copied." % task,
              file=sys.stderr)
        sys.exit(1)

    # Separate the normal detail logs from their summary files.
    detail_files = [f for f in new_files if not f.endswith('_summary.json')]
    summary_files = [f for f in new_files if f.endswith('_summary.json')]

    print(f"\nCopying {len(new_files)} new file(s) to {dest_dir}/ "
          f"(originals in {src_dir}/ left untouched):")

    # Keep track of the new name for each detail file so that its
    # matching summary file can use the same naming convention.
    renamed_by_stem = {}  
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
           
            renamed = base
        dest_path = os.path.join(dest_dir, renamed)
        shutil.copy2(f, dest_path)
        print(f"  {base}  ->  {dest_path}")


if __name__ == '__main__':
    main()
