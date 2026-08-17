import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from tot.models import claude_prompt


# ── shared helpers ───────────
def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = claude_prompt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    unique_ys = list(dict.fromkeys(ys))
    local_cache = {}

    def evaluate(y):
        return y, get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate, y): y for y in unique_ys}
        for future in as_completed(futures):
            y, value = future.result()
            local_cache[y] = value

    return [local_cache.get(y, 0) for y in ys]


def _merge_cascade_chain(chain):
    """
    Collapse one maximal run of cascaded β(c) calls (see collapse_cascades)
    into a single logical backtracking episode. Read-only: does not modify
    the raw event dicts in `chain`, only reads them.

    last['to_depth'] is always a real depth for every condition, including D
    — since D was revised to fall back to root/parent instead of aborting
    the puzzle, it no longer produces to_depth=None events. The None-handling
    below is kept defensively (harmless if ever exercised) but is not
    currently reachable by any condition.
    """
    first, last = chain[0], chain[-1]
    if last.get('to_depth') is None:
        jump = None
        degenerate = False
    else:
        jump = first['from_depth'] - last['to_depth']
        degenerate = jump == 1
    return {
        'origin_depth':          first['from_depth'],
        'final_target_depth':    last.get('to_depth'),
        'jump':                  jump,
        'degenerate':            degenerate,
        'cascade_chain_length':  len(chain),
        'reasons':               [e.get('reason') for e in chain],
        'raw_events':            chain,
    }


def collapse_cascades(backtracks):
    """
    Post-hoc, read-only regrouping of `backtracks` into logical episodes.
    Does NOT touch the input list/dicts and has no influence whatsoever on
    the search — it only re-reads what _beta()/_beta_crossword() already
    recorded.

    A "cascade chain" is a maximal run of consecutive raw events sharing the
    same cascade_id — an id assigned at recording time in _beta()/
    _beta_crossword(): a NEW id starts whenever a β(c) call is not a
    cascade (i.e. at least one new node-budget-consuming candidate was
    explored since the previous β(c) call, or this is the very first call),
    and consecutive calls keep the same id for as long as each one is a
    cascade (zero new candidates explored since the last call).

    Events without a 'cascade' key (e.g. parent-only backtracks, which are
    never produced by a β(c) model call at all) are each their own
    length-1 chain — there is no cascade concept for parent-only recovery,
    since it never skips a frame without trying its own next candidate.
    """
    if not backtracks:
        return []

    merged = []
    chain = [backtracks[0]]
    for event in backtracks[1:]:
        if event.get('cascade') and event.get('cascade_id') == chain[0].get('cascade_id'):
            chain.append(event)
        else:
            merged.append(_merge_cascade_chain(chain))
            chain = [event]
    merged.append(_merge_cascade_chain(chain))
    return merged


def summarize_backtracks(backtracks):
    """
    Aggregate stats over a run's backtrack log — both the raw event stream
    and (for conditions that use β(c)) a post-hoc cascade-collapsed logical
    view. Purely a reporting function: it never influences the search, and
    collapse_cascades() only reads the already-recorded raw events.

    A backtrack with jump == 1 is "degenerate": it moved exactly one level
    up, indistinguishable from what parent-only backtracking would have
    done at that point. Only jump > 1 events are evidence that non-parent
    (β(c)) backtracking actually diverged from the parent-only baseline —
    pct_jumps_gt1 / mean_jump_size quantify how much of the search's
    backtracking was genuinely non-parent vs. accidentally parent-only.

    num_backtracks / mean_jump_size / pct_jumps_gt1 are kept as the
    original keys (unchanged definition/values) for backward compatibility
    with Condition 1 (parent-only) callers that predate this cascade
    instrumentation; num_backtracks_raw / mean_jump_size_raw /
    pct_jumps_gt1_raw are identical aliases, added alongside the merged and
    cascade-specific stats below.

    beta_calls / num_cascaded_calls / pct_cascaded_calls /
    cascade_chain_length_histogram only count events with
    beta_model_call=True — i.e. actual β(c) model invocations
    (_beta()/_beta_crossword() reaching the point of calling the LLM), not
    every entry in `backtracks`. Parent-only conditions never set this flag,
    so these are correctly 0 for Condition 1.

    cascade_chain_length counts the number of raw β(c) MODEL CALLS composing
    a logical episode — not the number of stack frames skipped in between
    them. Skipped frames (where `if bt_depth < depth: return None, bt_depth`
    fires in _diligent_recurse / _diligent_recurse_crossword) never
    themselves call the model, so they aren't independently observable
    "calls" to count; only the β(c) calls that actually happen are counted.
    """
    if not backtracks:
        return {
            'num_backtracks': 0, 'mean_jump_size': 0.0, 'pct_jumps_gt1': 0.0,
            'num_backtracks_raw': 0, 'mean_jump_size_raw': 0.0, 'pct_jumps_gt1_raw': 0.0,
            'num_logical_backtracks': 0, 'mean_jump_size_merged': 0.0, 'pct_jumps_gt1_merged': 0.0,
            'beta_calls': 0, 'num_cascaded_calls': 0, 'pct_cascaded_calls': 0.0,
            'cascade_chain_length_histogram': {},
            'backtracks_merged': [],
            'num_unrecoverable': 0,
        }

    # Condition D (strict non-parent β(c)) can record events with jump=None
    # (unrecoverable branches — see _strict_nonparent). num_backtracks_raw
    # still counts every raw event (including unrecoverable ones), but the
    # numeric jump stats (mean/pct>1) only make sense over events that
    # actually have a jump, so those are computed over the filtered subset.
    # For Conditions A/B/C every event always has a real jump, so this
    # filter is a no-op for them.
    all_jumps = [b['jump'] for b in backtracks]
    jumps = [j for j in all_jumps if j is not None]
    non_degenerate = [j for j in jumps if j > 1]
    num_backtracks_raw = len(all_jumps)
    mean_jump_size_raw = (sum(jumps) / len(jumps)) if jumps else 0.0
    pct_jumps_gt1_raw = (len(non_degenerate) / len(jumps)) if jumps else 0.0
    num_unrecoverable = sum(1 for b in backtracks if b.get('unrecoverable'))

    merged = collapse_cascades(backtracks)
    all_m_jumps = [m['jump'] for m in merged]
    m_jumps = [j for j in all_m_jumps if j is not None]
    m_non_degenerate = [j for j in m_jumps if j > 1]
    num_logical_backtracks = len(all_m_jumps)
    mean_jump_size_merged = (sum(m_jumps) / len(m_jumps)) if m_jumps else 0.0
    pct_jumps_gt1_merged = (len(m_non_degenerate) / len(m_jumps)) if m_jumps else 0.0

    beta_events = [b for b in backtracks if b.get('beta_model_call')]
    beta_calls = len(beta_events)
    cascaded = sum(1 for b in beta_events if b.get('cascade'))
    pct_cascaded_calls = cascaded / beta_calls if beta_calls else 0.0

    beta_merged = collapse_cascades(beta_events)
    histogram = {}
    for m in beta_merged:
        L = m['cascade_chain_length']
        histogram[L] = histogram.get(L, 0) + 1

    return {
        'num_backtracks':  num_backtracks_raw,
        'mean_jump_size':  mean_jump_size_raw,
        'pct_jumps_gt1':   pct_jumps_gt1_raw,
        'num_backtracks_raw': num_backtracks_raw,
        'mean_jump_size_raw': mean_jump_size_raw,
        'pct_jumps_gt1_raw':  pct_jumps_gt1_raw,
        'num_logical_backtracks': num_logical_backtracks,
        'mean_jump_size_merged':  mean_jump_size_merged,
        'pct_jumps_gt1_merged':   pct_jumps_gt1_merged,
        'beta_calls':          beta_calls,
        'num_cascaded_calls':  cascaded,
        'pct_cascaded_calls':  pct_cascaded_calls,
        'cascade_chain_length_histogram': histogram,
        'backtracks_merged': merged,
        'num_unrecoverable': num_unrecoverable,
    }


def _is_terminal(task, proposal):
    """
    Terminal check for Game24 — this file only handles game24 now
    (crosswords has its own dfs_crossword.py). Delegates to
    task.is_terminal(), e.g. Game24Task: return 'left: 24' in y.
    """
    return task.is_terminal(proposal)


def get_proposals(task, x, y, n_propose=1):
    """
    Generate candidate next steps for Game24.

    If y is already a terminal state, return [] immediately so neither
    _recurse_parent nor _diligent_recurse recurses further and solicits
    phantom recap steps from the model.
    """
    if y and _is_terminal(task, y):
        return []

    propose_prompt = task.propose_prompt_wrap(x, y)
    raw_outputs = claude_prompt(propose_prompt, n=n_propose, stop=None)

    # Collect 'left:' lines across ALL n_propose outputs so that n_propose > 1
    # actually produces multiple distinct candidate branches.
    game24_proposals = []
    seen = set()
    for out in raw_outputs:
        for line in out.split('\n'):
            line = line.strip()
            if '=' in line and 'left:' in line and line not in seen:
                game24_proposals.append(line)
                seen.add(line)

    return [y + p + '\n' for p in game24_proposals]


# ── non-parent backtrack target selection ─────────────────────────────────────

BACKTRACK_PROMPT = '''You are solving the following problem:
{input}

A reasoning path has been explored step by step, and this branch has failed.

Here is the full reasoning path (each step builds on the previous):
{ancestors}
Step {fail_depth}: FAILED - incorrect or incomplete solution.

Your task is to identify β(c): the deepest recoverable prefix of this reasoning path. Do not simply identify the first incorrect statement — a step can be locally valid and still be a dead end if no continuation from it can reach a correct solution. Identify the deepest step from which a DIFFERENT continuation could still reach a correct solution; search will restart from that step. If no step beyond the immediate parent is recoverable, return the immediate parent. The immediate parent is the fallback and is always an acceptable answer when no deeper recoverable ancestor exists.

You may briefly reason about it, but your response MUST end with exactly one line in this exact format, with nothing else after it and no other digits on that line:
ANSWER: <step number>'''

_ANSWER_PATTERN = re.compile(r'ANSWER:\s*(\d+)')


def select_backtrack_target(x, ancestor_ys):
    """
    Present the full ancestor chain to the model and ask it to identify
    β(c) — the deepest recoverable prefix of the reasoning path (not
    merely the last locally-valid step; see BACKTRACK_PROMPT).
    Returns (selected_y, selected_depth, raw_output, parse_method).

    Prefers the structured "ANSWER: <n>" line. Falls back to the last bare
    number in the response only if that's missing — earlier versions always
    took the last bare number, which silently picked up the echoed
    `Step {fail_depth}: FAILED` label from the prompt/response instead of
    the model's actual answer whenever it added commentary after stating
    the number, producing a spurious jump == 0 (backtracking to the very
    step that just failed).
    """
    lines = []
    for i, y in enumerate(ancestor_ys):
        if y == '':
            lines.append(f"Step 0 (root): [start — problem not yet begun]")
        else:
            lines.append(f"Step {i}:\n{y.strip()}")

    prompt = BACKTRACK_PROMPT.format(
        input=x,
        ancestors='\n'.join(lines),
        fail_depth=len(ancestor_ys) - 1
    )

    output = claude_prompt(prompt, n=1, stop=None, max_tokens=2000)[0]

    match = _ANSWER_PATTERN.search(output)
    if match:
        target_depth = int(match.group(1))
        target_depth = max(0, min(target_depth, len(ancestor_ys) - 1))
        parse_method = 'structured'
    else:
        numbers = re.findall(r'\b(\d+)\b', output)
        if numbers:
            target_depth = max(0, min(int(numbers[-1]), len(ancestor_ys) - 1))
            parse_method = 'fallback_last_number'
        else:
            target_depth = max(0, len(ancestor_ys) - 2)  # fallback: parent
            parse_method = 'fallback_no_number'

    return ancestor_ys[target_depth], target_depth, output, parse_method


# ── Condition 1: Parent-Only Backtracking (baseline) ─────────────────────────

def _recurse_parent(task, x, idx, y, depth, T_max, v_th,
                    n_evaluate, n_propose, node_budget, info, to_print):
    """
    Recursive DFS — Condition 1 (parent-only recovery). Whenever this
    subtree reaches a dead end (no candidate survives pruning, or a
    committed candidate fails), control returns to the immediate parent —
    the recovery point is always exactly one level up, unlike
    _diligent_recurse, which invokes β(c) to select the recovery point.

    Fix 3 is applied upstream in get_proposals — if y is already terminal,
    proposals will be [] and this function returns None immediately, preventing
    phantom recap steps from being generated.

    Fix 1 is applied below — if a proposal is detected as terminal by _is_terminal,
    it is tested immediately before falling through to the depth-limit check.

    Fix 2 (subtree backtrack recording) is applied after each recursive call
    that returns None, so that pruning-based backtracks are counted correctly.
    """
    if node_budget[0] <= 0:
        return None

    proposals = get_proposals(task, x, y, n_propose=n_propose)
    if not proposals:
        # get_proposals returned [] — either y is already terminal (Fix 3)
        # or the model produced nothing. Either way, nothing to explore.
        return None

    values = get_values(task, x, proposals, n_evaluate)
    ranked = sorted(zip(proposals, values), key=lambda p: p[1], reverse=True)

    # ── shared instrumentation ─────────────────────────────────────────────
    # candidates_evaluated / nodes_explored / candidates_pruned are all
    # computed over the FULL ranked list up front, not incrementally inside
    # the loop below. The loop `return`s as soon as a solution is found, so
    # it can skip lower-ranked candidates that were nonetheless already
    # value-evaluated (and already scored pass/fail against v_th) by
    # get_values() above — counting incrementally in the loop would
    # undercount both nodes_explored and candidates_pruned whenever the
    # search short-circuits on a higher-ranked winner. This keeps
    # candidates_evaluated == nodes_explored + candidates_pruned true by
    # construction, and is identical in definition/increment-point to
    # _diligent_recurse below, fixing the prior nodes_explored asymmetry.
    n_pass = sum(1 for _, v in ranked if v > v_th)
    n_prune = len(ranked) - n_pass
    info['candidates_evaluated'] += len(ranked)
    info['nodes_explored'] += n_pass
    info['candidates_pruned'] += n_prune
    if to_print:
        print(f"{'  '*depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={n_pass} pruned={n_prune}")

    for proposal, value in ranked:
        if node_budget[0] <= 0:
            break

        # node_budget accounting is unchanged from before this instrumentation
        # fix — still consumed once per candidate considered here, pass or
        # fail — so budget-exhaustion timing is identical to prior behavior.
        node_budget[0] -= 1

        if value <= v_th:
            if to_print:
                print(f"{'  '*depth}[pruned] {proposal.strip()!r}  score={value:.3f}")
            continue

        if to_print:
            print(f"{'  '*depth}[d={depth+1}] {proposal.strip()!r}  score={value:.3f}")

        # ── Fix 1: catch terminal state (e.g. Game24 'left: 24') early ─────────
        # test_output verifies the Steps trace directly (see verify_steps in
        # game24.py) — no need to ask the model to regenerate a separate
        # 'Answer: ...' summary line, which risked hallucinating an
        # incorrect expression even when the original steps were valid.
        if _is_terminal(task, proposal):
            result = task.test_output(idx, proposal)
            if result['r'] == 1:
                if to_print:
                    print(f"\nIt SOLVED at depth {depth+1} (terminal state detected)")
                return proposal
            info['backtracks'].append({
                'from_depth': depth + 1,
                'to_depth':   depth,
                'jump':       1,
                'degenerate': True,
                'reason':     'terminal fail (wrong answer at left: 24)',
                'rejected_proposal': proposal
            })
            if to_print:
                print(f"{'  '*(depth+1)}[BACKTRACK parent] depth {depth+1} → {depth}")
            continue

        # ── Standard depth-limit terminal check ───────────────────────────────
        at_terminal = depth + 1 >= T_max
        if at_terminal:
            result = task.test_output(idx, proposal)
            if result['r'] == 1:
                if to_print:
                    print(f"\n✓ SOLVED at depth {depth+1}")
                return proposal
            info['backtracks'].append({
                'from_depth': depth + 1,
                'to_depth':   depth,
                'jump':       1,
                'degenerate': True,
                'reason':     'terminal fail (depth limit)',
                'rejected_proposal': proposal
            })
            if to_print:
                print(f"{'  '*(depth+1)}[BACKTRACK parent] depth {depth+1} → {depth}")
            continue

        # ── Recurse deeper ─────────────────────────────────────────────────────
        sol = _recurse_parent(task, x, idx, proposal, depth + 1,
                              T_max, v_th, n_evaluate, n_propose, node_budget, info, to_print)
        if sol is not None:
            return sol

        # ── Fix 2: record backtrack when subtree is exhausted ─────────────────
        # Previously missing — parent-only backtracks through pruning were
        # never counted, giving backtracks=0 even on puzzles where the model
        # explored and abandoned multiple branches.
        info['backtracks'].append({
            'from_depth': depth + 1,
            'to_depth':   depth,
            'jump':       1,
            'degenerate': True,
            'reason':     'subtree exhausted',
            'rejected_proposal': proposal
        })

    return None  # all children exhausted → implicit backtrack to caller


def solve_dfs(args, task, idx, to_print=True):
    """Parent-only backtracking DFS — baseline condition."""
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    x           = task.get_input(idx)
    T_max       = task.steps
    v_th        = getattr(args, 'v_th', 0.5)
    node_budget = [getattr(args, 'node_budget', 30)]
    n_propose   = getattr(args, 'n_generate_sample', 1)

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_parent'
    }

    sol = _recurse_parent(task, x, idx, '', 0, T_max, v_th,
                          args.n_evaluate_sample, n_propose, node_budget, info, to_print)
    info['solution'] = sol
    info.update(summarize_backtracks(info['backtracks']))
    ys = [sol] if sol else ['']
    return ys, {'steps': [], **info}


# ── Condition 2: Diligent Learner Non-Parent Backtracking ────────────────────
#
# Implements Shalev-Shwartz & Shashua (2025) "From Reasoning to Super-Intelligence"
#
# Two assumptions:
#   1. γ-GPAC: sample B candidate next steps at each node — at least one
#              is correct with probability γ  (B = n_generate_sample)
#   2. β(c) recovery: when all B candidates at a node fail — i.e. the
#              search has reached a dead end, whether by exhausting an
#              explored subtree or by every candidate being pruned — the
#              model identifies β(c) = the deepest recoverable prefix of
#              the ancestor chain (not merely the last locally-valid
#              step), and control jumps there directly, bypassing all
#              intermediate nodes (non-parent recovery).
#
# Structural difference vs parent-only:
#   Parent-only  → every dead end returns control exactly 1 level up (depth - 1)
#   Diligent     → every dead end invokes β(c); failure propagates up the
#                  call stack until depth β(c), which may skip many levels

def _diligent_recurse(task, x, idx, y, depth, ancestors,
                      T_max, B, v_th, n_evaluate, node_budget, info, to_print,
                      backtrack_fn=None):
    """
    Recursive Diligent Learner DFS.

    Fix 3 applies upstream in get_proposals — terminal states return [] so
    this function returns early without generating phantom steps.

    Fix 1 applies below — _is_terminal catches solved states before the
    depth-limit check, consistent with _recurse_parent.

    backtrack_fn: callable with the same interface as _beta() (ancestor
    chain in, target depth out) — the ONLY thing that differs between
    Condition B (β(c), backtrack_fn=_beta, the default) and Condition C
    (fixed k=2, backtrack_fn=_fixed_k2). Everything else in this function —
    candidate generation, scoring, pruning, node-budget accounting,
    recursion order — is identical for both conditions, since they call
    this exact same function. Defaults to _beta so existing callers
    (solve_dfs_nonparent) are behaviorally unchanged.

    Returns
    -------
    (solution_str, None)      — a correct answer was found
    (None, backtrack_depth)   — subtree failed; caller should absorb if
                                backtrack_depth >= caller's depth,
                                or propagate upward if backtrack_depth < caller's depth
    """
    if backtrack_fn is None:
        backtrack_fn = _beta

    if node_budget[0] <= 0:
        return None, max(0, depth - 1)

    # ── Assumption 1: γ-GPAC — sample B candidate next steps ─────────────────
    proposals = get_proposals(task, x, y, n_propose=B)
    if not proposals:
        # get_proposals returned [] — y is already terminal (Fix 3) or no output.
        # Dead end: signal failure upward so caller can try siblings or
        # invoke β(c) to select the recovery point.
        bt_depth = backtrack_fn(x, ancestors + [y], depth, info, to_print, reason='no proposals / already terminal')
        return None, bt_depth

    values = get_values(task, x, proposals, n_evaluate)
    ranked = sorted(zip(proposals, values), key=lambda pv: pv[1], reverse=True)
    valid  = [(p, v) for p, v in ranked if v > v_th]

    # ── shared instrumentation: identical definition/increment point to
    # _recurse_parent above — computed over the full ranked list up front,
    # not incrementally in the loop below (which can return early once a
    # solution is found), so candidates_evaluated == nodes_explored +
    # candidates_pruned holds by construction.
    info['candidates_evaluated'] += len(ranked)
    info['nodes_explored'] += len(valid)
    info['candidates_pruned'] += len(ranked) - len(valid)
    if to_print:
        print(f"{'  '*depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={len(valid)} pruned={len(ranked) - len(valid)}")

    if not valid:
        # ── Dead end: every candidate pruned by the value threshold ──────────
        # (Assumption 2) β(c) selects the recovery point — the deepest
        # recoverable prefix — rather than just returning to the parent.
        bt_depth = backtrack_fn(x, ancestors + [y], depth, info, to_print, reason='all pruned')
        return None, bt_depth

    new_ancestors = ancestors + [y]

    for proposal, value in valid:
        if node_budget[0] <= 0:
            break

        # node_budget accounting is unchanged from before this instrumentation
        # fix — still consumed only by candidates that already passed the
        # threshold (this loop only ever sees `valid` candidates), so
        # budget-exhaustion timing is identical to prior behavior.
        node_budget[0] -= 1
        # cascade-detection bookkeeping only (see _beta) — increments exactly
        # once per candidate actually committed to here, independent of the
        # publicly-reported nodes_explored (which is computed in one batch
        # per node, up front, so it can't tell us in real time whether THIS
        # specific candidate was tried before the next β(c) call).
        info['_live_explore_counter'] = info.get('_live_explore_counter', 0) + 1

        if to_print:
            print(f"{'  '*depth}[d={depth+1}] {proposal.strip()[:80]!r}  score={value:.3f}")

        # ── Fix 1: catch terminal state (e.g. Game24 'left: 24') early ─────────
        # test_output verifies the Steps trace directly (see verify_steps in
        # game24.py) — no separate 'Answer: ...' regeneration call needed.
        if _is_terminal(task, proposal):
            result = task.test_output(idx, proposal)
            if result['r'] == 1:
                if to_print:
                    print(f"\nSOLVED at depth {depth+1} (terminal state detected)")
                return proposal, None

            dead_chain = new_ancestors + [proposal]
            bt_depth = backtrack_fn(x, dead_chain, depth + 1, info, to_print, reason='terminal fail (wrong answer at left: 24)')
            if bt_depth < depth:
                return None, bt_depth
            continue

        # ── Standard depth-limit terminal check ───────────────────────────────
        at_terminal = depth + 1 >= T_max
        if at_terminal:
            result = task.test_output(idx, proposal)
            if result['r'] == 1:
                if to_print:
                    print(f"\n✓ SOLVED at depth {depth+1}")
                return proposal, None

            dead_chain = new_ancestors + [proposal]
            bt_depth = backtrack_fn(x, dead_chain, depth + 1, info, to_print, reason='terminal fail (depth limit)')
            if bt_depth < depth:
                return None, bt_depth
            continue

        # ── Non-terminal: recurse into this candidate ─────────────────────────
        sol, bt_depth = _diligent_recurse(
            task, x, idx, proposal, depth + 1,
            new_ancestors, T_max, B, v_th, n_evaluate,
            node_budget, info, to_print, backtrack_fn=backtrack_fn
        )

        if sol is not None:
            return sol, None            # ── success propagates up ──

        if bt_depth is not None and bt_depth < depth:
            # β(c) is above me — pass the failure upward without trying siblings
            return None, bt_depth

        # β(c) >= depth → the failure is absorbed here; try next sibling

    # ── Dead end: every candidate at this node has been tried and failed ─────
    # (subtree exhausted) β(c) selects the recovery point.
    bt_depth = backtrack_fn(x, new_ancestors, depth, info, to_print, reason='all candidates exhausted')
    return None, bt_depth


def _beta(x, ancestor_chain, from_depth, info, to_print, reason=''):
    """
    Identify β(c): the deepest recoverable prefix of the ancestor chain —
    the point search should restart from, not merely the last locally-valid
    step. Calls the LLM to locate this recovery point.
    Records the backtrack event and returns the target depth.

    The early return below (len(ancestor_chain) <= 1) does NOT call the
    model and does NOT append to info['backtracks'] — it's deliberately not
    tagged beta_model_call, so it's correctly excluded from beta_calls /
    cascade stats in summarize_backtracks (see docstring there).

    Cascade tagging: a call is a "cascade" iff, since the previous ACTUAL
    β(c) model call in this puzzle, zero new node-budget-consuming
    candidates were explored — detected by comparing info's
    '_live_explore_counter' (incremented in _diligent_recurse's loop, see
    there) against the snapshot taken at the previous β(c) call. This is
    read from the *actual* recursion/search state (the real commit-point
    counter), not inferred from depth numbers.
    """
    if len(ancestor_chain) <= 1:
        return 0  # can only go back to root

    _, bt_depth, raw_output, parse_method = select_backtrack_target(x, ancestor_chain)
    jump = from_depth - bt_depth
    degenerate = jump == 1

    explore_count = info.get('_live_explore_counter', 0)
    prev_snapshot = info.get('_explore_snapshot_at_last_beta')
    explored_since_previous = None if prev_snapshot is None else explore_count - prev_snapshot
    cascade = (prev_snapshot is not None) and (explore_count == prev_snapshot)
    if cascade:
        info['_cascade_position'] = info.get('_cascade_position', 1) + 1
    else:
        info['_cascade_id_counter'] = info.get('_cascade_id_counter', 0) + 1
        info['_cascade_position'] = 1
    cascade_id = info['_cascade_id_counter']
    cascade_position = info['_cascade_position']
    info['_explore_snapshot_at_last_beta'] = explore_count

    info['backtracks'].append({
        'from_depth':      from_depth,
        'to_depth':        bt_depth,
        'jump':            jump,
        'degenerate':      degenerate,
        'reason':          reason,
        'rejected_proposal': ancestor_chain[-1],
        'beta_raw_response': raw_output,
        'beta_parse_method': parse_method,
        'beta_model_call':  True,
        'cascade':          cascade,
        'cascade_id':       cascade_id,
        'cascade_position': cascade_position,
        'explored_since_previous_beta': explored_since_previous,
        'parent_fallback':  jump == 1,
    })

    if to_print:
        tag = ' [degenerate: same as parent-only]' if degenerate else ''
        cascade_tag = ' [CASCADE]' if cascade else ''
        print(f"{'  '*from_depth}[β(c) BACKTRACK] {reason} — "
              f"depth {from_depth} → {bt_depth}  (skipped {jump - 1} levels){tag}{cascade_tag}")

    return bt_depth


# ── Condition C: Fixed k=2 Backtracking (deterministic baseline) ─────────────

def _fixed_k2(x, ancestor_chain, from_depth, info, to_print, reason='', k=2):
    """
    Deterministic fixed-jump backtrack target selection — Condition C.
    Drop-in replacement for _beta() (identical call signature, passed as
    _diligent_recurse's backtrack_fn), so the search machinery around it —
    candidate generation, scoring, pruning, node-budget accounting,
    recursion order — is byte-for-byte identical to Condition B. The ONLY
    difference is how the target depth is chosen: no LLM call, no prompt,
    just target_depth = max(0, from_depth - k).

    Mirrors _beta's root-clamp convention exactly: len(ancestor_chain) <= 1
    means there is nothing above the root to jump to, so it returns 0
    without recording an event, same as _beta. The max(0, from_depth - k)
    formula already naturally clamps near-root cases correctly on its own
    (e.g. from_depth=1 -> max(0, 1-2)=0, jump=1, never jump=2 when only one
    ancestor level actually exists) — no separate special-casing needed.

    beta_model_call is explicitly set to False (not merely omitted) to
    distinguish "this condition could in principle have used a model-call
    mechanism but deliberately doesn't" from parent-only's backtrack events,
    which never carry the key at all because the concept doesn't apply
    there architecturally. summarize_backtracks()'s beta_calls/cascade
    stats only count beta_model_call=True events, so Condition C correctly
    reports beta_calls=0 and contributes nothing to cascade statistics —
    cascade is a β(c)-specific concept (see collapse_cascades docstring)
    and is deliberately not computed for fixed-jump backtracking at all;
    no cascade/cascade_id/cascade_position/explored_since_previous_beta
    keys are set on these events.
    """
    if len(ancestor_chain) <= 1:
        return 0  # can only go back to root — same convention as _beta

    bt_depth = max(0, from_depth - k)
    jump = from_depth - bt_depth
    degenerate = jump == 1

    info['backtracks'].append({
        'from_depth':      from_depth,
        'to_depth':        bt_depth,
        'jump':            jump,
        'degenerate':      degenerate,
        'reason':          reason,
        'rejected_proposal': ancestor_chain[-1],
        'beta_model_call':  False,
    })

    if to_print:
        tag = ' [degenerate: same as parent-only]' if degenerate else ''
        print(f"{'  '*from_depth}[FIXED-k={k} BACKTRACK] {reason} — "
              f"depth {from_depth} → {bt_depth}  (skipped {jump - 1} levels){tag}")

    return bt_depth


def solve_dfs_nonparent(args, task, idx, to_print=True):
    """
    Diligent Learner non-parent backtracking DFS — Condition 2.

    Implements the inference procedure from:
      Shalev-Shwartz & Shashua (2025) "From Reasoning to Super-Intelligence:
      A Search-Theoretic Perspective"

    Key properties:
    - At each node, samples B candidate next steps (γ-GPAC assumption)
    - At every dead end (all candidates pruned, or an explored subtree
      exhausted), invokes β(c) to select the recovery point — the deepest
      recoverable prefix of the ancestor chain — not necessarily the
      immediate parent
    - Failure propagates up the recursive call stack, bypassing intermediate
      nodes, until it reaches the β(c) recovery point
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    x           = task.get_input(idx)
    T_max       = task.steps
    v_th        = getattr(args, 'v_th', 0.5)
    B           = getattr(args, 'n_generate_sample', 3)
    node_budget = [getattr(args, 'node_budget', 100)]

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_diligent'
    }

    sol, _ = _diligent_recurse(
        task, x, idx, '', 0, [],
        T_max, B, v_th, args.n_evaluate_sample,
        node_budget, info, to_print
    )

    info['solution'] = sol
    info.update(summarize_backtracks(info['backtracks']))
    for k in [k for k in info if k.startswith('_')]:
        del info[k]
    ys = [sol] if sol else ['']
    return ys, {'steps': [], **info}


def solve_dfs_fixed_k2(args, task, idx, to_print=True):
    """
    Fixed k=2 backtracking DFS — Condition C (deterministic baseline).

    Identical search machinery to solve_dfs_nonparent (Condition B): same
    candidate generation (get_proposals), same scoring (get_values), same
    pruning (v_th), same node-budget accounting, same recursion
    (_diligent_recurse) — reused directly, not duplicated. The ONLY
    difference is the backtrack_fn passed to _diligent_recurse: _fixed_k2
    (target_depth = max(0, from_depth - 2), no LLM call) instead of _beta
    (LLM-selected target).

    Intended to answer: does intelligent β(c) target selection provide an
    advantage over simply making a fixed jump of k=2, holding everything
    else in the search constant?
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    x           = task.get_input(idx)
    T_max       = task.steps
    v_th        = getattr(args, 'v_th', 0.5)
    B           = getattr(args, 'n_generate_sample', 3)
    node_budget = [getattr(args, 'node_budget', 100)]

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_fixed_k2'
    }

    sol, _ = _diligent_recurse(
        task, x, idx, '', 0, [],
        T_max, B, v_th, args.n_evaluate_sample,
        node_budget, info, to_print, backtrack_fn=_fixed_k2
    )

    info['solution'] = sol
    info.update(summarize_backtracks(info['backtracks']))
    ys = [sol] if sol else ['']
    return ys, {'steps': [], **info}


# ── Condition D: Strict Non-Parent β(c) (never falls back to parent) ──────────
#
# The LLM identifies the deepest recoverable ancestor, exactly as in
# Condition B, EXCEPT the immediate parent is NEVER a legal target once
# from_depth >= 2. D never aborts the whole puzzle and never selects the
# parent: when no legal non-parent ancestor exists (from_depth <= 1), it
# falls back deterministically to root; when legal non-parent ancestors
# exist (from_depth >= 2) but the model returns UNRECOVERABLE, an
# out-of-range depth, or the parent itself, it falls back to the
# deterministic strict-non-parent target from_depth - 2 (the deepest legal
# non-parent ancestor) — never to the parent. See _strict_nonparent's
# docstring.

STRICT_NONPARENT_BACKTRACK_PROMPT = '''You are solving the following problem:
{input}

A reasoning path has been explored step by step, and this branch has failed.

Here is the full reasoning path (each step builds on the previous):
{ancestors}
Step {fail_depth}: FAILED - incorrect or incomplete solution.

Your task is to identify the deepest recoverable NON-PARENT prefix of this
reasoning path.

The immediate parent (Step {fail_depth} - 1) is NOT an allowed target in this
condition.

A step is recoverable if a DIFFERENT continuation from that step could still
reach a correct solution. A step can be locally valid and still be
unrecoverable if every possible continuation from it leads to failure.

Among the ancestors strictly above the immediate parent, identify the
DEEPEST step from which a different continuation could still reach a correct
solution.

If no recoverable non-parent ancestor exists, the branch is UNRECOVERABLE.
Do NOT return the immediate parent in that case.

Your response MUST end with exactly one line in one of these formats:

ANSWER: <step number>

or

UNRECOVERABLE

There must be nothing after that final line.'''

_STRICT_ANSWER_PATTERN = re.compile(r'ANSWER:\s*(\d+)')
_STRICT_UNRECOVERABLE_PATTERN = re.compile(r'\bUNRECOVERABLE\b', re.IGNORECASE)


def _parse_strict_response(output, from_depth):
    """
    Parse a Condition D response. Returns (target_depth, parse_method,
    unrecoverable). target_depth is None iff unrecoverable is True.

    Deliberately does NOT reuse select_backtrack_target's fallback-to-last-
    bare-number logic — that fallback exists in B to recover a plausible
    answer from a slightly malformed response, but for D any ambiguity must
    resolve to "unrecoverable", never to a guessed number that could
    accidentally equal the forbidden parent depth.
    """
    match = _STRICT_ANSWER_PATTERN.search(output)
    if match:
        target_depth = int(match.group(1))
        # Legal range for a non-parent target: 0 <= target_depth <= from_depth - 2.
        if 0 <= target_depth <= from_depth - 2:
            return target_depth, 'structured', False
        return None, 'structured_rejected_invalid_target', True
    if _STRICT_UNRECOVERABLE_PATTERN.search(output):
        return None, 'structured_unrecoverable', True
    return None, 'unparseable_defaulted_unrecoverable', True


def _strict_nonparent(x, ancestor_chain, from_depth, info, to_print, reason=''):
    """
    Condition D backtrack target selection — non-parent PREFERRED, and the
    immediate parent is NEVER a legal target once from_depth >= 2. Same call
    signature as _beta()/_fixed_k2() (drop-in backtrack_fn for
    _diligent_recurse).

    Every path through this function returns a REAL depth >= 0. There is no
    "unrecoverable, abort the whole puzzle" sentinel: D never terminates a
    puzzle's search early, and it never falls back to the parent. The two
    cases:

      1. No legal non-parent ancestor exists (from_depth <= 1): fall back to
         root (target_depth=0) WITHOUT calling the model — the legal
         non-parent set {0 .. from_depth-2} is empty by construction, so
         there is nothing to ask about. This is NOT a failure state; DFS
         simply continues from the root's own remaining candidates.
      2. Legal non-parent targets exist (from_depth >= 2). The model may
         select any 0 <= target <= from_depth-2. If it returns a valid one,
         use it. If it returns UNRECOVERABLE, an out-of-range depth, or the
         forbidden parent (from_depth-1) itself, use the deterministic
         strict-non-parent fallback target_depth = from_depth - 2 — the
         deepest legal non-parent ancestor. This keeps the invariant
         target_depth <= from_depth - 2 (hence target_depth != from_depth-1)
         true for every from_depth >= 2 event, regardless of what the model
         says.

    Because every return value is a real depth, the caller's existing
    `if bt_depth < depth: return None, bt_depth` propagation logic in
    _diligent_recurse — completely unmodified — does the actual fallback for
    us: returning depth 0 (case 1) or from_depth-2 (case 2 fallback) is
    itself sufficient for normal upward propagation to land exactly there,
    the same mechanism B and C already rely on.
    """
    if from_depth <= 1:
        target_depth = 0
        jump = from_depth - target_depth
        info['backtracks'].append({
            'from_depth': from_depth,
            'to_depth': target_depth,
            'jump': jump,
            'degenerate': jump == 1,
            'reason': reason,
            'rejected_proposal': ancestor_chain[-1] if ancestor_chain else None,
            'beta_model_call': False,
            'strict_nonparent': True,
            'parent_forbidden': True,
            'no_legal_nonparent': True,
            'fallback_to_root': True,
            'nonparent_target_selected': False,
            'model_unrecoverable': False,
            'invalid_target': False,
            'fallback_to_deepest_nonparent': False,
        })
        if to_print:
            print(f"{'  '*from_depth}[STRICT-D ROOT-FALLBACK] {reason} — "
                  f"depth {from_depth}: no non-parent ancestor exists, falling back to root")
        return target_depth

    lines = []
    for i, y in enumerate(ancestor_chain):
        if y == '':
            lines.append("Step 0 (root): [start — problem not yet begun]")
        else:
            lines.append(f"Step {i}:\n{y.strip()}")

    prompt = STRICT_NONPARENT_BACKTRACK_PROMPT.format(
        input=x,
        ancestors='\n'.join(lines),
        fail_depth=from_depth
    )
    output = claude_prompt(prompt, n=1, stop=None, max_tokens=2000)[0]

    target_depth, parse_method, unrecoverable = _parse_strict_response(output, from_depth)

    if unrecoverable:
        # Legal non-parent targets existed, but the model didn't give us a
        # usable one — use the deterministic strict-non-parent fallback
        # (deepest legal non-parent ancestor), NEVER the immediate parent,
        # and never abort the puzzle. from_depth - 2 always differs from
        # the forbidden parent (from_depth - 1) by construction.
        nonparent_fallback_depth = from_depth - 2
        info['backtracks'].append({
            'from_depth': from_depth,
            'to_depth': nonparent_fallback_depth,
            'jump': from_depth - nonparent_fallback_depth,  # always 2
            'degenerate': False,
            'reason': reason,
            'rejected_proposal': ancestor_chain[-1],
            'beta_raw_response': output,
            'beta_parse_method': parse_method,
            'beta_model_call': True,
            'strict_nonparent': True,
            'parent_forbidden': True,
            'no_legal_nonparent': False,
            'fallback_to_root': False,
            'nonparent_target_selected': False,
            'model_unrecoverable': parse_method in ('structured_unrecoverable',
                                                      'unparseable_defaulted_unrecoverable'),
            'invalid_target': parse_method == 'structured_rejected_invalid_target',
            'fallback_to_deepest_nonparent': True,
        })
        if to_print:
            print(f"{'  '*from_depth}[STRICT-D NONPARENT-FALLBACK] {reason} — "
                  f"depth {from_depth} → {nonparent_fallback_depth}  (parse: {parse_method})")
        return nonparent_fallback_depth

    jump = from_depth - target_depth
    info['backtracks'].append({
        'from_depth': from_depth,
        'to_depth': target_depth,
        'jump': jump,
        'degenerate': jump == 1,
        'reason': reason,
        'rejected_proposal': ancestor_chain[-1],
        'beta_raw_response': output,
        'beta_parse_method': parse_method,
        'beta_model_call': True,
        'strict_nonparent': True,
        'parent_forbidden': True,
        'no_legal_nonparent': False,
        'fallback_to_root': False,
        'nonparent_target_selected': True,
        'model_unrecoverable': False,
        'invalid_target': False,
        'fallback_to_deepest_nonparent': False,
    })
    if to_print:
        print(f"{'  '*from_depth}[STRICT-D BACKTRACK] {reason} — "
              f"depth {from_depth} → {target_depth}  (skipped {jump - 1} levels)")
    return target_depth


def solve_dfs_nonparent_strict(args, task, idx, to_print=True):
    """
    Strict non-parent β(c) DFS — Condition D. Never selects the immediate
    parent once from_depth >= 2, and never aborts the search.

    Identical search machinery to solve_dfs_nonparent (Condition B) and
    solve_dfs_fixed_k2 (Condition C): same candidate generation, scoring,
    pruning, node-budget accounting, recursion (_diligent_recurse) — reused
    directly, not duplicated. The ONLY difference is the backtrack_fn:
    _strict_nonparent, which forbids the immediate parent as an LLM target
    and falls back to root (no legal non-parent ancestor exists, from_depth
    <= 1) or to the deterministic strict-non-parent target from_depth - 2
    (model returns UNRECOVERABLE/invalid/parent, from_depth >= 2) — never
    to the parent, and never aborting the search for the whole puzzle.
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    x           = task.get_input(idx)
    T_max       = task.steps
    v_th        = getattr(args, 'v_th', 0.5)
    B           = getattr(args, 'n_generate_sample', 3)
    node_budget = [getattr(args, 'node_budget', 100)]

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_nonparent_strict'
    }

    sol, _ = _diligent_recurse(
        task, x, idx, '', 0, [],
        T_max, B, v_th, args.n_evaluate_sample,
        node_budget, info, to_print, backtrack_fn=_strict_nonparent
    )

    info['solution'] = sol
    info.update(summarize_backtracks(info['backtracks']))
    ys = [sol] if sol else ['']
    return ys, {'steps': [], **info}
