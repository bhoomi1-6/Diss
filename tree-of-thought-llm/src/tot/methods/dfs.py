import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from tot.models import claude_prompt


# Shared helpers to evaluate and rank the candidate next steps. 
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
    """Merges a run of cascaded beta(c) calls into one backtracking episode. Doesn't change the input."""
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
    Groups backtrack events into logical episodes after the search is done.
    Doesn't change the search itself.

    A cascade chain is a run of events sharing the same cascade_id (a new
    id starts whenever a beta(c) call isn't a cascade). Events with no
    'cascade' key, like parent-only backtracks, are each their own chain.
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
    Computes summary stats for one puzzle's backtrack log: raw counts plus
    (for beta(c) conditions) the cascade-collapsed view. Doesn't affect
    the search.

    jump == 1 means the backtrack went to the parent, same as condition A
    would do. jump > 1 means it skipped ahead. beta_calls/
    num_cascaded_calls/pct_cascaded_calls only count real LLM calls
    (beta_model_call=True), so they're 0 for condition A.
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

    # jump is None for unrecoverable branches, so stats use only the rest.
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
    """Terminal check for Game24 delegates to task.is_terminal()."""
    return task.is_terminal(proposal)


def get_proposals(task, x, y, n_propose=1):
    """
    Generate candidate next steps for Game24. Returns [] immediately if y
    is already terminal, so callers don't recurse further on a solved state.
    """
    if y and _is_terminal(task, y):
        return []

    propose_prompt = task.propose_prompt_wrap(x, y)
    raw_outputs = claude_prompt(propose_prompt, n=n_propose, stop=None)

    # Collect 'left:' lines from every output so multiple samples give
    # multiple candidate branches.
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
    Asks the model to pick beta(c), the deepest recoverable step in the
    ancestor chain (see BACKTRACK_PROMPT). Returns (selected_y,
    selected_depth, raw_output, parse_method).

    Prefers the "ANSWER: <n>" line falls back to the last number in the
    response if that's missing.
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


# Condition 1: Parent-Only Backtracking (baseline) 

def _recurse_parent(task, x, idx, y, depth, T_max, v_th,
                    n_evaluate, n_propose, node_budget, info, to_print):
    """
    Condition A: parent-only DFS. Every dead end returns to the immediate
    parent, one level up, unlike _diligent_recurse, which asks beta(c)
    where to recover to.
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

    # candidates_evaluated == nodes_explored + candidates_pruned holds by construction

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

        # node_budget is consumed once per candidate considered here, pass
        # or fail.
        node_budget[0] -= 1

        if value <= v_th:
            if to_print:
                print(f"{'  '*depth}[pruned] {proposal.strip()!r}  score={value:.3f}")
            continue

        if to_print:
            print(f"{'  '*depth}[d={depth+1}] {proposal.strip()!r}  score={value:.3f}")

        # Catch terminal state (e.g. Game24 'left: 24') early.
        # verifies the Steps trace directly (see verify_steps in game24.py).
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
                print(f"{'  '*(depth+1)}[BACKTRACK parent] depth {depth+1} -> {depth}")
            continue

        # Standard depth-limit terminal check
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
                print(f"{'  '*(depth+1)}[BACKTRACK parent] depth {depth+1} -> {depth}")
            continue

        # Recurse deeper
        sol = _recurse_parent(task, x, idx, proposal, depth + 1,
                              T_max, v_th, n_evaluate, n_propose, node_budget, info, to_print)
        if sol is not None:
            return sol

        # Record backtrack when this subtree is exhausted.
        info['backtracks'].append({
            'from_depth': depth + 1,
            'to_depth':   depth,
            'jump':       1,
            'degenerate': True,
            'reason':     'subtree exhausted',
            'rejected_proposal': proposal
        })

    return None  # all children exhausted, implicit backtrack made to caller


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


# Condition 2: Diligent Learner Non-Parent Backtracking 
#
# 1. Sample B candidate next steps at each node (B = n_generate_sample).
# 2. When every candidate at a node fails, ask the model for beta(c) - the
#    deepest recoverable step in the ancestor chain - and jump straight
#    there instead of just going up one level.

def _diligent_recurse(task, x, idx, y, depth, ancestors,
                      T_max, B, v_th, n_evaluate, node_budget, info, to_print,
                      backtrack_fn=None):
    """
    Recursive DFS shared by Conditions B, C, and D.

    backtrack_fn takes the ancestor chain and returns a target depth - it's
    the only thing that changes between conditions (_beta for B, _fixed_k2
    for C, _strict_nonparent for D). Everything else in this function is
    the same for all three.

    Returns (solution, None) if solved, or (None, backtrack_depth) if this
    subtree failed - the caller keeps propagating upward while
    backtrack_depth is above its own depth.
    """
    if backtrack_fn is None:
        backtrack_fn = _beta

    if node_budget[0] <= 0:
        return None, max(0, depth - 1)

    # Sample B candidate next steps
    proposals = get_proposals(task, x, y, n_propose=B)
    if not proposals:
        # No candidates: dead end. Ask backtrack_fn for a recovery point.
        bt_depth = backtrack_fn(x, ancestors + [y], depth, info, to_print, reason='no proposals / already terminal')
        return None, bt_depth

    values = get_values(task, x, proposals, n_evaluate)
    ranked = sorted(zip(proposals, values), key=lambda pv: pv[1], reverse=True)
    valid  = [(p, v) for p, v in ranked if v > v_th]

    # Counted up front, same as _recurse_parent, so candidates_evaluated
    # always equals nodes_explored + candidates_pruned.
    info['candidates_evaluated'] += len(ranked)
    info['nodes_explored'] += len(valid)
    info['candidates_pruned'] += len(ranked) - len(valid)
    if to_print:
        print(f"{'  '*depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={len(valid)} pruned={len(ranked) - len(valid)}")

    if not valid:
        # Every candidate was pruned. Ask backtrack_fn where to recover to.
        bt_depth = backtrack_fn(x, ancestors + [y], depth, info, to_print, reason='all pruned')
        return None, bt_depth

    new_ancestors = ancestors + [y]

    for proposal, value in valid:
        if node_budget[0] <= 0:
            break

        # Budget is only spent on candidates that passed pruning.
        node_budget[0] -= 1
        # Separate counter for cascade detection in _beta - nodes_explored
        # is only updated once per node, so this tracks progress live.
        info['_live_explore_counter'] = info.get('_live_explore_counter', 0) + 1

        if to_print:
            print(f"{'  '*depth}[d={depth+1}] {proposal.strip()[:80]!r}  score={value:.3f}")

        # Catch terminal state (e.g. Game24 'left: 24') early
        # verifies the Steps trace directly.
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

        # Standard depth-limit terminal check
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

        # Non-terminal: recurse into this candidate 
        sol, bt_depth = _diligent_recurse(
            task, x, idx, proposal, depth + 1,
            new_ancestors, T_max, B, v_th, n_evaluate,
            node_budget, info, to_print, backtrack_fn=backtrack_fn
        )

        if sol is not None:
            return sol, None         #if successful, return solution up the chain

        if bt_depth is not None and bt_depth < depth:
            # Target is above me then pass the failure up without trying siblings
            return None, bt_depth

        # Target is at or below me then try the next sibling

    # All candidates tried and failed. Ask backtrack_fn where to recover to.
    bt_depth = backtrack_fn(x, new_ancestors, depth, info, to_print, reason='all candidates exhausted')
    return None, bt_depth


def _beta(x, ancestor_chain, from_depth, info, to_print, reason=''):
    """
    Asks the model for beta(c), the deepest recoverable step in the
    ancestor chain, records the backtrack event, and returns that depth.

    If there's only the root to go back to, returns 0 without calling the
    model (not counted as a real beta(c) call).

    A call counts as a "cascade" if no new candidates were explored since
    the last beta(c) call - tracked with a live counter.
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
        print(f"{'  '*from_depth}[beta(c) BACKTRACK] {reason} — "
              f"depth {from_depth} -> {bt_depth}  (skipped {jump - 1} levels){tag}{cascade_tag}")

    return bt_depth


# Condition C: Fixed k=2 Backtracking (deterministic baseline) 

def _fixed_k2(x, ancestor_chain, from_depth, info, to_print, reason='', k=2):
    """
    Condition C: always jumps back exactly k levels, no LLM call.
    target_depth = max(0, from_depth - k).

    Same call signature as _beta() so it can be used as backtrack_fn.
    beta_model_call is set to False so summarize_backtracks() correctly
    reports 0 beta calls for this condition.
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
              f"depth {from_depth} -> {bt_depth}  (skipped {jump - 1} levels){tag}")

    return bt_depth


def solve_dfs_nonparent(args, task, idx, to_print=True):
    """
    Condition B: non-parent backtracking DFS. At every dead end, asks the
    model for beta(c) - the deepest recoverable step - instead of just
    going back one level.
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
    Condition C: same search as solve_dfs_nonparent (Condition B), but
    uses _fixed_k2 instead of _beta, so backtracking always jumps 2 levels
    with no model call.
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


# Condition D: Strict Non-Parent beta(c) (never falls back to parent) 
#
# Same as Condition B, but the immediate parent is never a legal target
# once from_depth >= 2. Falls back to root if there's no legal non-parent
# ancestor, or to from_depth - 2 if the model doesn't give a usable answer.

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
    Parses a Condition D response. Returns (target_depth, parse_method,
    unrecoverable) - target_depth is None only when unrecoverable is True.

    Doesn't fall back to guessing a number like select_backtrack_target
    does, since a wrong guess could land on the forbidden parent depth.
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
    Condition D: picks a non-parent backtrack target. The parent is never
    a legal answer once from_depth >= 2. Always returns a real depth,
    never aborts.

    If from_depth <= 1 there's no non-parent ancestor to pick, so it falls
    back to root without asking the model. Otherwise the model picks a
    depth between 0 and from_depth-2; if it fails to give a usable one,
    falls back to from_depth - 2.
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
                  f"depth {from_depth} -> {nonparent_fallback_depth}  (parse: {parse_method})")
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
              f"depth {from_depth} -> {target_depth}  (skipped {jump - 1} levels)")
    return target_depth


def solve_dfs_nonparent_strict(args, task, idx, to_print=True):
    """
    Condition D: same search as Condition B, but uses _strict_nonparent so
    backtracking never lands on the immediate parent.
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
