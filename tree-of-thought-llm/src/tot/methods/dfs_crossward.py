import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tot.models import claude_prompt
from tot.prompts.crosswords import propose_prompt
from tot.methods.dfs import summarize_backtracks

# ── Crossword-specific Condition 1 (parent-only backtracking) DFS ────────────

CONFIDENCE_TO_VALUE = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}
_LINE_PATTERN = re.compile(r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$')


def _prompt_wrap(obs):
    return propose_prompt.format(input=obs)


def _parse_line(line):
    match = _LINE_PATTERN.match(line)
    return match.groups() if match else None


def _parse_response(response):
    results = []
    for line in response.split('\n'):
        parsed = _parse_line(line)
        if parsed is None:
            continue
        pos, word, confidence = parsed
        results.append((pos.lower() + '. ' + word.lower(), CONFIDENCE_TO_VALUE.get(confidence, 0)))
    return results or None


def _get_candidates_to_scores(env, n_propose):
    obs = env.render()
    if obs in env.cache:
        return env.cache[obs]

    # n_propose separate completions, same token/dollar cost as before — just
    # fired concurrently instead of sequentially (mirrors get_values() in
    # dfs.py), so wall-clock time doesn't scale linearly with n_propose.
    prompt = _prompt_wrap(obs)
    with ThreadPoolExecutor(max_workers=n_propose) as executor:
        responses = [r for r in executor.map(lambda _: claude_prompt(prompt, n=1)[0], range(n_propose))]
    candidates_to_scores = {}
    for response in responses:
        parsed = _parse_response(response)
        if parsed:
            for candidate, score in parsed:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score

    env.cache[obs] = candidates_to_scores
    return candidates_to_scores


def select_final_state(node_snapshots, method='deepest'):
    """
    Select which explored node_snapshots entry is reported as the puzzle's
    final board.

    method='deepest' (MAIN A/B/C/D selection): the state reached at the
    greatest search depth (len(n['actions'])), first-encountered on ties.
    Does NOT use r_word.

    method='best_r_word' (preserved, NOT used by the main experiment): the
    state with the highest ground-truth-informed r_word — the original
    ToT paper's "+best state" oracle ablation. r_word is recorded on every
    snapshot regardless of which method is selected, so it stays available
    for that separate analysis later.
    """
    if method == 'deepest':
        return max(node_snapshots, key=lambda n: len(n['actions']))
    elif method == 'best_r_word':
        return max(node_snapshots, key=lambda n: n['r_word'])
    raise ValueError(f"Unknown selection method: {method!r}")


def _depth_stats(node_snapshots, selected):
    """Depth bookkeeping for the concise summary — read-only, no effect on selection."""
    depths = [len(n['actions']) for n in node_snapshots]
    max_depth = max(depths)
    tie = depths.count(max_depth) > 1
    selected_depth = len(selected['actions'])
    return max_depth, selected_depth, tie


def build_concise_summary(info, args, idx, usage_this_puzzle):
    """
    Build a flat, pandas-friendly per-puzzle summary dict. Purely
    instrumentation/output — reads already-computed info/usage fields and
    invents nothing; does not affect search behavior. Detailed logs (the
    full `info` dict, dumped by run.py) are unaffected and still written
    in full alongside this.
    """
    candidates_evaluated = info.get('candidates_evaluated', 0)
    candidates_pruned = info.get('candidates_pruned', 0)
    candidates_passed = info.get('nodes_explored', 0)
    pruning_rate = (candidates_pruned / candidates_evaluated) if candidates_evaluated > 0 else None

    solution = info.get('solution')
    summary = {
        'puzzle_id': idx,
        'condition': args.method_search,
        'solved': bool(solution['r_game']) if solution else False,
        'r_game': solution['r_game'] if solution else None,
        'r_word': solution['r_word'] if solution else None,
        'r_letter': solution['r_letter'] if solution else None,

        'nodes_explored': info.get('nodes_explored'),
        'candidates_evaluated': candidates_evaluated,
        'candidates_passed': candidates_passed,
        'candidates_pruned': candidates_pruned,
        'pruning_rate': pruning_rate,

        'backtracks': info.get('num_backtracks'),
        'logical_backtracks': info.get('num_logical_backtracks'),
        'mean_jump': info.get('mean_jump_size'),
        'pct_jumps_gt1': info.get('pct_jumps_gt1'),

        'max_depth': info.get('max_depth'),
        'selected_depth': info.get('selected_depth'),
        'selection_method': info.get('selection_method'),

        'completion_tokens': usage_this_puzzle.get('completion_tokens'),
        'prompt_tokens': usage_this_puzzle.get('prompt_tokens'),
        'total_tokens': (usage_this_puzzle.get('completion_tokens', 0)
                          + usage_this_puzzle.get('prompt_tokens', 0)),
        'cost': usage_this_puzzle.get('cost'),
    }

    # beta(c)/non-parent-specific fields. beta_calls is 0 for Condition A
    # (parent-only) and Condition C (fixed k=2, no LLM call) without any
    # special-casing, since summarize_backtracks() only counts events with
    # beta_model_call=True and those conditions never set that flag.
    if args.method_search in ('dfs_crossword_nonparent', 'dfs_crossword_nonparent_strict',
                               'dfs_crossword_fixed_k2'):
        summary['beta_calls'] = info.get('beta_calls', 0)
        summary['cascade_calls'] = info.get('num_cascaded_calls', 0)
        summary['cascade_percentage'] = info.get('pct_cascaded_calls', 0.0)
        summary['logical_backtracks'] = info.get('num_logical_backtracks')

    return summary


def _recurse_crossword(env, actions, node_budget, node_snapshots, info,
                        depth, prune, max_per_state, n_propose, to_print):
    if node_budget[0] <= 0:
        return

    candidates_to_scores = _get_candidates_to_scores(env, n_propose)
    if not candidates_to_scores:
        return

    board, status, steps = env.board.copy(), env.status.copy(), env.steps
    ranked = sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
    # candidates_evaluated counts every distinct word this node's propose call
    # scored, regardless of whether it later got committed to — the crossword
    # analogue of Game24's get_values() step (here, the LLM's confidence vote
    # *is* the evaluation, there's no separate scalar-threshold pass). It's
    # recorded up front, unconditional of how the loop below exits, so
    # candidates_evaluated == nodes_explored (this node's share) +
    # candidates_pruned holds regardless of early breaks — same invariant as
    # dfs.py's _recurse_parent/_diligent_recurse.
    info['candidates_evaluated'] += len(ranked)
    cnt_per_state = 0

    for action in ranked:
        if node_budget[0] <= 0:
            break

        _, _, _, step_info = env.step(action)

        # not violating any existing (already-filled) constraint — deliberately
        # does NOT check env.steps < 10 here: that check used to be combined
        # with this one, which meant the action that fills the 10th and final
        # slot (env.steps going 9 -> 10) always failed this gate and the whole
        # block below — including node_snapshots.append(), the one place a
        # completed board's r_game gets recorded — was skipped entirely, so a
        # perfectly solved board could never be reported as solved. Recursion
        # (which genuinely should stop once the board is full) is gated
        # separately below, after the snapshot is recorded.
        if node_budget[0] > 0 and not any(s == 2 for s in env.status):
            # Cap check happens BEFORE incrementing cnt_per_state (fixed
            # off-by-one): previously cnt_per_state was bumped for the
            # capped-out candidate too, so it was silently excluded from
            # BOTH nodes_explored (correctly) AND candidates_pruned
            # (incorrectly, since candidates_pruned = len(ranked) -
            # cnt_per_state used the inflated cnt_per_state). Checking the
            # cap first means cnt_per_state ends the loop equal to the
            # number of candidates actually explored at this node, so the
            # capped-out candidate now correctly falls into
            # len(ranked) - cnt_per_state. Explored candidates, their
            # order, and node_budget consumption are unchanged — this is
            # instrumentation-only.
            if cnt_per_state >= max_per_state:
                env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
                break
            cnt_per_state += 1

            node_budget[0] -= 1
            info['nodes_explored'] += 1
            count = env.prompt_status()
            actions.append(action)

            if to_print:
                print(f"{'  ' * (depth + 1)}[d={depth + 1}] {action!r}  "
                      f"r_word={step_info['r_word']:.2f}  {count}")

            node_snapshots.append({
                'actions': actions.copy(),
                'board': ''.join(env.board),
                'r_word': step_info['r_word'],
                'r_letter': step_info['r_letter'],
                'r_game': step_info['r_game'],
            })

            if env.steps < 10:
                if not prune or count['impossible'] < 1:
                    _recurse_crossword(env, actions, node_budget, node_snapshots, info,
                                        depth + 1, prune, max_per_state, n_propose, to_print)
                    reason = 'subtree exhausted'
                else:
                    reason = f"pruned: {count['impossible']} word(s) marked impossible"
            else:
                reason = 'board complete (10/10 slots filled)'

            info['backtracks'].append({
                'from_depth': depth + 1, 'to_depth': depth, 'jump': 1,
                'degenerate': True, 'reason': reason,
            })
            actions.pop()

        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)

    info['candidates_pruned'] += len(ranked) - cnt_per_state
    if to_print:
        print(f"{'  ' * depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={cnt_per_state} pruned={len(ranked) - cnt_per_state}")


def solve_dfs_crossword(args, task, idx, to_print=True):
    """Parent-only backtracking DFS for MiniCrosswords — Condition 1.

    Always backtracks exactly one level (to the parent board state) and
    tries the next-ranked candidate word, mirroring _recurse_parent in
    dfs.py but operating on task.env's board/status instead of a text y.
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    env = task.env
    env.reset(idx)

    node_budget = [getattr(args, 'node_budget', 100)]
    max_per_state = getattr(args, 'max_per_state', 3)
    n_propose = getattr(args, 'n_generate_sample', 8)
    prune = not getattr(args, 'no_prune', False)
    selection_method = getattr(args, 'selection_method', 'deepest')

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_crossword_parent',
    }
    node_snapshots = []

    _recurse_crossword(env, [], node_budget, node_snapshots, info,
                        0, prune, max_per_state, n_propose, to_print)

    info.update(summarize_backtracks(info['backtracks']))

    if node_snapshots:
        best = select_final_state(node_snapshots, method=selection_method)
        info['solution'] = best
        info['selection_method'] = selection_method
        max_depth, selected_depth, tie = _depth_stats(node_snapshots, best)
        info['max_depth'] = max_depth
        info['selected_depth'] = selected_depth
        info['selection_tie'] = tie
        board_str = best['board']
    else:
        info['selection_method'] = selection_method
        info['max_depth'] = None
        info['selected_depth'] = None
        info['selection_tie'] = None
        board_str = ''.join(env.board)

    y = '\n'.join(' '.join(board_str[i * 5:(i + 1) * 5]) for i in range(5))

    if to_print:
        print(y)
        print('best r_word:', info['solution']['r_word'] if info['solution'] else 0)

    return [y], info



# ── shared β(c) machinery (crossword-specific) ───────────────────────────────

CROSSWORD_BACKTRACK_PROMPT = '''You are solving the following mini crossword:
{input}

A search path has filled words step by step, and this branch has failed
(hit an unsatisfiable board state, or every continuation from here was
pruned/exhausted).

Here is the full fill history:
{ancestors}
Step {fail_depth}: FAILED - this branch cannot be completed correctly.

Your task is to identify β(c): the deepest recoverable prefix of this fill
history. Do not simply identify the first word that looks locally
plausible — a filled word can be individually valid (matches its own
clue) and still be a dead end if it forecloses every crossing word needed
elsewhere on the board. Identify the deepest step from which a DIFFERENT
word choice could still lead to a fully correct board; search will
restart from that step. If no step beyond the immediate parent is
recoverable, return the immediate parent. The immediate parent is the
fallback and is always an acceptable answer when no deeper recoverable
ancestor exists.

You may briefly reason about it, but your response MUST end with exactly
one line in this exact format, with nothing else after it and no other
digits on that line:
ANSWER: <step number>'''

_ANSWER_PATTERN = re.compile(r'ANSWER:\s*(\d+)')


def _beta_crossword(x, actions, board_history, info, to_print, reason=''):
    """
    Crossword analogue of _beta() in dfs.py. actions/board_history describe
    the current path (Step i+1 = actions[i], board after it =
    board_history[i]); from_depth is derived as len(actions) so callers
    don't have to track it separately — matching the invariant in dfs.py
    that from_depth == len(ancestor_chain) - 1.

    IMPORTANT: call this BEFORE popping the just-failed action off
    actions/board_history when the dead end is a specific candidate
    (pruned / board-complete cases) — this mirrors dead_chain =
    new_ancestors + [proposal] in dfs.py, so the failed step is itself
    part of what the model sees and can be selected as the recovery point.
    For 'all candidates exhausted' / 'no candidates', call it with actions
    already back to this frame's own depth (nothing extra appended).
    """
    from_depth = len(actions)
    if from_depth == 0:
        return 0  # can only go back to root

    lines = ["Step 0 (root): [empty board, no words filled]"]
    for i, (action, board) in enumerate(zip(actions, board_history)):
        grid = '\n'.join(' '.join(board[r * 5:(r + 1) * 5]) for r in range(5))
        lines.append(f"Step {i + 1}: filled {action}\nBoard:\n{grid}")

    prompt = CROSSWORD_BACKTRACK_PROMPT.format(
        input=x,
        ancestors='\n'.join(lines),
        fail_depth=from_depth
    )
    output = claude_prompt(prompt, n=1, stop=None, max_tokens=2000)[0]

    match = _ANSWER_PATTERN.search(output)
    if match:
        target_depth = max(0, min(int(match.group(1)), from_depth))
        parse_method = 'structured'
    else:
        numbers = re.findall(r'\b(\d+)\b', output)
        if numbers:
            target_depth = max(0, min(int(numbers[-1]), from_depth))
            parse_method = 'fallback_last_number'
        else:
            target_depth = max(0, from_depth - 1)  # fallback: parent
            parse_method = 'fallback_no_number'

    jump = from_depth - target_depth
    degenerate = jump == 1

    # cascade detection — identical mechanism to _beta() in dfs.py: a call is
    # a cascade iff zero new node-budget-consuming candidates were explored
    # (per '_live_explore_counter', incremented in _diligent_recurse_crossword's
    # commit point) since the previous actual β(c) model call.
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
        'from_depth': from_depth,
        'to_depth': target_depth,
        'jump': jump,
        'degenerate': degenerate,
        'reason': reason,
        'rejected_action': actions[-1] if actions else None,
        'beta_raw_response': output,
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
        print(f"{'  ' * from_depth}[β(c) BACKTRACK] {reason} — "
              f"depth {from_depth} → {target_depth}  (skipped {jump - 1} levels){tag}{cascade_tag}")

    return target_depth


# ── Condition C: Fixed k=2 Backtracking (deterministic baseline, crossword) ──

def _fixed_k2_crossword(x, actions, board_history, info, to_print, reason='', k=2):
    """
    Crossword analogue of _fixed_k2() in dfs.py — same drop-in-replacement
    role for _beta_crossword() that _fixed_k2 plays for _beta(). Deterministic
    target_depth = max(0, from_depth - k), no LLM call, no cascade concept
    (see _fixed_k2's docstring in dfs.py for the full rationale, identical
    here). from_depth derived the same way _beta_crossword does
    (len(actions)), so the depth semantics match exactly.
    """
    from_depth = len(actions)
    if from_depth == 0:
        return 0  # can only go back to root — same convention as _beta_crossword

    target_depth = max(0, from_depth - k)
    jump = from_depth - target_depth
    degenerate = jump == 1

    info['backtracks'].append({
        'from_depth': from_depth,
        'to_depth': target_depth,
        'jump': jump,
        'degenerate': degenerate,
        'reason': reason,
        'rejected_action': actions[-1] if actions else None,
        'beta_model_call': False,
    })

    if to_print:
        tag = ' [degenerate: same as parent-only]' if degenerate else ''
        print(f"{'  ' * from_depth}[FIXED-k={k} BACKTRACK] {reason} — "
              f"depth {from_depth} → {target_depth}  (skipped {jump - 1} levels){tag}")

    return target_depth


# ── Condition 2: Diligent Learner Non-Parent Backtracking (crossword) ────────

def _diligent_recurse_crossword(env, x, actions, board_history, node_budget, node_snapshots, info,
                                 depth, prune, max_per_state, n_propose, to_print,
                                 backtrack_fn=None):
    """
    β(c) non-parent backtracking DFS — crossword Condition 2. Mirrors
    _diligent_recurse in dfs.py: every dead end (no candidates, pruned, or
    board-complete/exhausted) invokes β(c) instead of always returning to
    the immediate parent. There's no "solved, stop early" branch here —
    same as C1, crossword search is anytime; node_snapshots are scored by
    r_word at the very end in solve_dfs_crossword_nonparent.

    Returns the backtrack target depth for this subtree. Each frame
    restores its own pre-loop env state (board/status/steps) before
    returning, so by the time a returned bt_depth reaches the frame that
    owns it, env is already sitting at that ancestor's exact state —
    no separate snapshot stack needed to "jump" multiple levels.

    backtrack_fn: same role as in dfs.py's _diligent_recurse — the ONLY
    thing that differs between crossword Condition B (_beta_crossword, the
    default) and Condition C (_fixed_k2_crossword). Everything else here —
    candidate generation, scoring, pruning, node-budget accounting,
    recursion order — is identical for both, since they call this exact
    same function.
    """
    if backtrack_fn is None:
        backtrack_fn = _beta_crossword

    if node_budget[0] <= 0:
        return max(0, depth - 1)

    candidates_to_scores = _get_candidates_to_scores(env, n_propose)
    if not candidates_to_scores:
        return backtrack_fn(x, actions, board_history, info, to_print,
                             reason='no candidates proposed')

    board, status, steps = env.board.copy(), env.status.copy(), env.steps
    ranked = sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
    # see matching comment in _recurse_crossword — recorded up front so the
    # invariant holds regardless of how the loop below exits (early beta
    # return, max_per_state cap, or budget exhaustion).
    info['candidates_evaluated'] += len(ranked)
    cnt_per_state = 0

    for action in ranked:
        if node_budget[0] <= 0:
            break

        _, _, _, step_info = env.step(action)

        if node_budget[0] > 0 and not any(s == 2 for s in env.status):
            # Cap check happens BEFORE incrementing cnt_per_state (fixed
            # off-by-one): previously cnt_per_state was bumped for the
            # capped-out candidate too, so it was silently excluded from
            # BOTH nodes_explored (correctly) AND candidates_pruned
            # (incorrectly, since candidates_pruned = len(ranked) -
            # cnt_per_state used the inflated cnt_per_state). Checking the
            # cap first means cnt_per_state ends the loop equal to the
            # number of candidates actually explored at this node, so the
            # capped-out candidate now correctly falls into
            # len(ranked) - cnt_per_state. Explored candidates, their
            # order, and node_budget consumption are unchanged — this is
            # instrumentation-only.
            if cnt_per_state >= max_per_state:
                env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
                break
            cnt_per_state += 1

            node_budget[0] -= 1
            info['nodes_explored'] += 1
            # cascade-detection bookkeeping only (see _beta_crossword) — a
            # dedicated counter, kept separate from nodes_explored so the
            # mechanism matches dfs.py's exactly and doesn't depend on
            # nodes_explored's own accounting semantics staying incremental.
            info['_live_explore_counter'] = info.get('_live_explore_counter', 0) + 1
            count = env.prompt_status()
            actions.append(action)
            board_history.append(''.join(env.board))

            if to_print:
                print(f"{'  ' * (depth + 1)}[d={depth + 1}] {action!r}  "
                      f"r_word={step_info['r_word']:.2f}  {count}")

            node_snapshots.append({
                'actions': actions.copy(),
                'board': ''.join(env.board),
                'r_word': step_info['r_word'],
                'r_letter': step_info['r_letter'],
                'r_game': step_info['r_game'],
            })

            if env.steps < 10:
                if not prune or count['impossible'] < 1:
                    bt_depth = _diligent_recurse_crossword(
                        env, x, actions, board_history, node_budget, node_snapshots, info,
                        depth + 1, prune, max_per_state, n_propose, to_print,
                        backtrack_fn=backtrack_fn)
                else:
                    bt_depth = backtrack_fn(
                        x, actions, board_history, info, to_print,
                        reason=f"pruned: {count['impossible']} word(s) marked impossible")
            else:
                bt_depth = backtrack_fn(
                    x, actions, board_history, info, to_print,
                    reason='board complete (10/10 slots filled)')

            actions.pop()
            board_history.pop()
            env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)

            if bt_depth < depth:
                # β(c) points above this node — this frame's own state is
                # already restored above; propagate upward without trying
                # remaining siblings.
                info['candidates_pruned'] += len(ranked) - cnt_per_state
                if to_print:
                    print(f"{'  ' * depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
                          f"passed={cnt_per_state} pruned={len(ranked) - cnt_per_state}")
                return bt_depth
            continue  # bt_depth >= depth: absorbed here, try next candidate

        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)

    info['candidates_pruned'] += len(ranked) - cnt_per_state
    if to_print:
        print(f"{'  ' * depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={cnt_per_state} pruned={len(ranked) - cnt_per_state}")
    return backtrack_fn(x, actions, board_history, info, to_print,
                         reason='all candidates exhausted')


def solve_dfs_crossword_nonparent(args, task, idx, to_print=True):
    """Diligent Learner (β(c)) non-parent backtracking DFS — crossword Condition 2."""
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    env = task.env
    env.reset(idx)

    node_budget = [getattr(args, 'node_budget', 100)]
    max_per_state = getattr(args, 'max_per_state', 3)
    n_propose = getattr(args, 'n_generate_sample', 8)
    prune = not getattr(args, 'no_prune', False)
    selection_method = getattr(args, 'selection_method', 'deepest')

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_crossword_diligent',
    }
    node_snapshots = []

    x = env.render()  # clue/puzzle description shown to the β(c) model

    _diligent_recurse_crossword(env, x, [], [], node_budget, node_snapshots, info,
                                 0, prune, max_per_state, n_propose, to_print)

    info.update(summarize_backtracks(info['backtracks']))
    for k in [k for k in info if k.startswith('_')]:
        del info[k]

    if node_snapshots:
        best = select_final_state(node_snapshots, method=selection_method)
        info['solution'] = best
        info['selection_method'] = selection_method
        max_depth, selected_depth, tie = _depth_stats(node_snapshots, best)
        info['max_depth'] = max_depth
        info['selected_depth'] = selected_depth
        info['selection_tie'] = tie
        board_str = best['board']
    else:
        info['selection_method'] = selection_method
        info['max_depth'] = None
        info['selected_depth'] = None
        info['selection_tie'] = None
        board_str = ''.join(env.board)

    y = '\n'.join(' '.join(board_str[i * 5:(i + 1) * 5]) for i in range(5))

    if to_print:
        print(y)
        print('best r_word:', info['solution']['r_word'] if info['solution'] else 0)

    return [y], info


def solve_dfs_crossword_fixed_k2(args, task, idx, to_print=True):
    """
    Fixed k=2 backtracking DFS for MiniCrosswords — Condition C.

    Identical search machinery to solve_dfs_crossword_nonparent (Condition
    B): same candidate generation, scoring, pruning, node-budget accounting,
    recursion (_diligent_recurse_crossword) — reused directly, not
    duplicated. The ONLY difference is the backtrack_fn passed in:
    _fixed_k2_crossword (target_depth = max(0, from_depth - 2), no LLM
    call) instead of _beta_crossword.
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    env = task.env
    env.reset(idx)

    node_budget = [getattr(args, 'node_budget', 100)]
    max_per_state = getattr(args, 'max_per_state', 3)
    n_propose = getattr(args, 'n_generate_sample', 8)
    prune = not getattr(args, 'no_prune', False)
    selection_method = getattr(args, 'selection_method', 'deepest')

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_crossword_fixed_k2',
    }
    node_snapshots = []

    x = env.render()

    _diligent_recurse_crossword(env, x, [], [], node_budget, node_snapshots, info,
                                 0, prune, max_per_state, n_propose, to_print,
                                 backtrack_fn=_fixed_k2_crossword)

    info.update(summarize_backtracks(info['backtracks']))

    if node_snapshots:
        best = select_final_state(node_snapshots, method=selection_method)
        info['solution'] = best
        info['selection_method'] = selection_method
        max_depth, selected_depth, tie = _depth_stats(node_snapshots, best)
        info['max_depth'] = max_depth
        info['selected_depth'] = selected_depth
        info['selection_tie'] = tie
        board_str = best['board']
    else:
        info['selection_method'] = selection_method
        info['max_depth'] = None
        info['selected_depth'] = None
        info['selection_tie'] = None
        board_str = ''.join(env.board)

    y = '\n'.join(' '.join(board_str[i * 5:(i + 1) * 5]) for i in range(5))

    if to_print:
        print(y)
        print('best r_word:', info['solution']['r_word'] if info['solution'] else 0)

    return [y], info


# ── Condition D: Strict Non-Parent β(c) (crossword) ───────────────────────────

STRICT_NONPARENT_CROSSWORD_BACKTRACK_PROMPT = '''You are solving the following mini crossword:
{input}

A search path has filled words step by step, and this branch has failed
(hit an unsatisfiable board state, or every continuation from here was
pruned/exhausted).

Here is the full fill history:
{ancestors}
Step {fail_depth}: FAILED - this branch cannot be completed correctly.

Your task is to identify the deepest recoverable NON-PARENT prefix of this
fill history.

The immediate parent (Step {fail_depth} - 1) is NOT an allowed target in this
condition.

A step is recoverable if a DIFFERENT word choice from that step's completed
board could still lead to a fully correct board, respecting the crossing
constraints with every other word already filled at that point. A filled
word can be individually valid (matches its own clue) and still be
unrecoverable if every possible continuation from it — given the crossing
letters it locks in — leads to failure.

Among the ancestors strictly above the immediate parent, identify the
DEEPEST step from which a different word choice could still reach a fully
correct board.

If no recoverable non-parent ancestor exists, the branch is UNRECOVERABLE.
Do NOT return the immediate parent in that case.

Your response MUST end with exactly one line in one of these formats:

ANSWER: <step number>

or

UNRECOVERABLE

There must be nothing after that final line.'''

_STRICT_ANSWER_PATTERN_CROSSWORD = re.compile(r'ANSWER:\s*(\d+)')
_STRICT_UNRECOVERABLE_PATTERN_CROSSWORD = re.compile(r'\bUNRECOVERABLE\b', re.IGNORECASE)


def _parse_strict_response_crossword(output, from_depth):
    """Crossword analogue of _parse_strict_response() in dfs.py — identical logic."""
    match = _STRICT_ANSWER_PATTERN_CROSSWORD.search(output)
    if match:
        target_depth = int(match.group(1))
        if 0 <= target_depth <= from_depth - 2:
            return target_depth, 'structured', False
        return None, 'structured_rejected_invalid_target', True
    if _STRICT_UNRECOVERABLE_PATTERN_CROSSWORD.search(output):
        return None, 'structured_unrecoverable', True
    return None, 'unparseable_defaulted_unrecoverable', True


def _strict_nonparent_crossword(x, actions, board_history, info, to_print, reason=''):
    """
    Crossword analogue of _strict_nonparent() in dfs.py — non-parent
    PREFERRED backtracking. The immediate parent is NEVER a legal target
    once from_depth >= 2, and D never aborts the puzzle. See
    _strict_nonparent's docstring in dfs.py for the full rationale;
    identical here, using from_depth = len(actions) exactly like
    _beta_crossword.
    """
    from_depth = len(actions)

    if from_depth <= 1:
        target_depth = 0
        jump = from_depth - target_depth
        info['backtracks'].append({
            'from_depth': from_depth,
            'to_depth': target_depth,
            'jump': jump,
            'degenerate': jump == 1,
            'reason': reason,
            'rejected_action': actions[-1] if actions else None,
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
            print(f"{'  ' * from_depth}[STRICT-D ROOT-FALLBACK] {reason} — "
                  f"depth {from_depth}: no non-parent ancestor exists, falling back to root")
        return target_depth

    lines = ["Step 0 (root): [empty board, no words filled]"]
    for i, (action, board) in enumerate(zip(actions, board_history)):
        grid = '\n'.join(' '.join(board[r * 5:(r + 1) * 5]) for r in range(5))
        lines.append(f"Step {i + 1}: filled {action}\nBoard:\n{grid}")

    prompt = STRICT_NONPARENT_CROSSWORD_BACKTRACK_PROMPT.format(
        input=x,
        ancestors='\n'.join(lines),
        fail_depth=from_depth
    )
    output = claude_prompt(prompt, n=1, stop=None, max_tokens=2000)[0]

    target_depth, parse_method, unrecoverable = _parse_strict_response_crossword(output, from_depth)

    if unrecoverable:
        # Legal non-parent targets existed, but the model didn't give us a
        # usable one — use the deterministic strict-non-parent fallback
        # (deepest legal non-parent ancestor), NEVER the immediate parent.
        nonparent_fallback_depth = from_depth - 2
        info['backtracks'].append({
            'from_depth': from_depth,
            'to_depth': nonparent_fallback_depth,
            'jump': from_depth - nonparent_fallback_depth,  # always 2
            'degenerate': False,
            'reason': reason,
            'rejected_action': actions[-1] if actions else None,
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
            print(f"{'  ' * from_depth}[STRICT-D NONPARENT-FALLBACK] {reason} — "
                  f"depth {from_depth} → {nonparent_fallback_depth}  (parse: {parse_method})")
        return nonparent_fallback_depth

    jump = from_depth - target_depth
    info['backtracks'].append({
        'from_depth': from_depth,
        'to_depth': target_depth,
        'jump': jump,
        'degenerate': jump == 1,
        'reason': reason,
        'rejected_action': actions[-1] if actions else None,
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
        print(f"{'  ' * from_depth}[STRICT-D BACKTRACK] {reason} — "
              f"depth {from_depth} → {target_depth}  (skipped {jump - 1} levels)")
    return target_depth


def solve_dfs_crossword_nonparent_strict(args, task, idx, to_print=True):
    """
    Strict non-parent β(c) DFS for MiniCrosswords — Condition D. Never
    selects the immediate parent once from_depth >= 2, and never aborts
    the search.

    Identical search machinery to solve_dfs_crossword_nonparent (Condition
    B) and solve_dfs_crossword_fixed_k2 (Condition C): same candidate
    generation, scoring, pruning, node-budget accounting, recursion
    (_diligent_recurse_crossword) — reused directly, not duplicated. The
    ONLY difference is the backtrack_fn: _strict_nonparent_crossword, which
    forbids the parent as an LLM target and falls back to root (no legal
    non-parent ancestor, from_depth<=1) or the deterministic
    strict-non-parent target from_depth-2 (model returns
    UNRECOVERABLE/invalid/parent, from_depth>=2) — never to the parent,
    and never aborting the search for the whole puzzle.
    """
    global claude_prompt
    claude_prompt = partial(claude_prompt, temperature=args.temperature)
    print(claude_prompt)

    env = task.env
    env.reset(idx)

    node_budget = [getattr(args, 'node_budget', 100)]
    max_per_state = getattr(args, 'max_per_state', 3)
    n_propose = getattr(args, 'n_generate_sample', 8)
    prune = not getattr(args, 'no_prune', False)
    selection_method = getattr(args, 'selection_method', 'deepest')

    info = {
        'nodes_explored':       0,
        'candidates_evaluated': 0,
        'candidates_pruned':    0,
        'backtracks':           [],
        'solution':             None,
        'method':               'dfs_crossword_nonparent_strict',
    }
    node_snapshots = []

    x = env.render()

    _diligent_recurse_crossword(env, x, [], [], node_budget, node_snapshots, info,
                                 0, prune, max_per_state, n_propose, to_print,
                                 backtrack_fn=_strict_nonparent_crossword)

    info.update(summarize_backtracks(info['backtracks']))

    if node_snapshots:
        best = select_final_state(node_snapshots, method=selection_method)
        info['solution'] = best
        info['selection_method'] = selection_method
        max_depth, selected_depth, tie = _depth_stats(node_snapshots, best)
        info['max_depth'] = max_depth
        info['selected_depth'] = selected_depth
        info['selection_tie'] = tie
        board_str = best['board']
    else:
        info['selection_method'] = selection_method
        info['max_depth'] = None
        info['selected_depth'] = None
        info['selection_tie'] = None
        board_str = ''.join(env.board)

    y = '\n'.join(' '.join(board_str[i * 5:(i + 1) * 5]) for i in range(5))

    if to_print:
        print(y)
        print('best r_word:', info['solution']['r_word'] if info['solution'] else 0)

    return [y], info