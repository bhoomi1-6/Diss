# FINAL-experiment log naming convention

Applies only to files under `logs/final_experiment/`. Pilot/dev logs under
`logs/game24/` and `logs/crosswords/` keep run.py's own existing filenames
unchanged — this convention is applied by `scripts/run_final_experiment.py`
when it copies a run's output into the final-experiment tree, never by
editing run.py itself.

## Pattern

```
<benchmark>_<condition_dir>_<backend>_B<n_generate_sample>_E<n_evaluate_sample>_budget<node_budget>_vth<v_th>[_maxstate<max_per_state>_prune<True|False>]_<start>_<end>_<timestamp>.json
```

The `maxstate`/`prune` segment is included only for `crosswords` (Game24
never reads those args, even though they're always present in `config`
since they're global CLI flags).

## Example

```
game24_A_parent_bedrock_B3_E1_budget30_vth0.5_900_1000_20260817_120000.json
crosswords_D_strict_nonparent_bedrock_B5_E1_budget100_vth0.5_maxstate3_pruneTrue_0_20_20260817_120500.json
```

## Condition directory names

| `--method_search` value | `condition_dir` |
|---|---|
| `dfs`, `dfs_crossword` | `A_parent` |
| `dfs_nonparent`, `dfs_crossword_nonparent` | `B_beta_c` |
| `dfs_fixed_k2`, `dfs_crossword_fixed_k2` | `C_fixed_k2` |
| `dfs_nonparent_strict`, `dfs_crossword_nonparent_strict` | `D_strict_nonparent` |

## Where every value comes from

Every field is read directly out of the `config` dict embedded in the
detail log itself (`info['config'] = vars(args)`, written by run.py on
every puzzle) — never re-derived, parsed from the original filename, or
guessed. This matters because run.py's own filenames don't embed every
field consistently (e.g. Condition A's Game24 filename never includes
`n_generate_sample` at all) — reading `config` is the only way to recover
the true value in every case. The timestamp is the one exception: it isn't
stored in the JSON body, so it's read from run.py's own generated
filename.

Crossword conditions also produce a sibling `..._summary.json` (the
concise per-puzzle summary from `build_concise_summary`); it is renamed to
match its paired detail file's new name with `_summary.json` appended, so
the two always sort next to each other.
