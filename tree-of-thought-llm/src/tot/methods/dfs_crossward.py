import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tot.models import claude_prompt
from tot.prompts.crosswords import propose_prompt
from tot.methods.dfs import summarize_backtracks


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

    # Generate several independent candidate solutions in parallel from the
    # current board state. Parallel generation reduces the time required to
    # obtain the set of proposals used to rank possible next moves.
    prompt = _prompt_wrap(obs)
    with ThreadPoolExecutor(max_workers=n_propose) as executor:
        responses = [r for r in executor.map(lambda _: claude_prompt(prompt, n=1)[0], range(n_propose))]

    # Parse the model responses and combine scores for candidates that appear in
    # multiple responses. A higher total score indicates stronger agreement
    # across the generated proposals.
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
    Picks which explored state is reported as the puzzle's final board.

    'deepest' (used by the main experiment): the state reached at the
    greatest depth, first on ties. Doesn't look at r_word.

    'best_r_word': the state with the highest ground-truth r_word - the
    original ToT paper's oracle ablation, kept but not used.
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
    """Builds a flat per-puzzle summary dict for pandas. Doesn't affect the search or the full info dict written by run.py."""
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

    # beta(c)/non-parent-specific fields, beta_calls  is 0 for Condition A
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

    info['candidates_evaluated'] += len(ranked)
    cnt_per_state = 0

    for action in ranked:
        if node_budget[0] <= 0:
            break

        _, _, _, step_info = env.step(action)

        if node_budget[0] > 0 and not any(s == 2 for s in env.status):
           # Limit the number of candidates explored from a single board state.
           # The counter is updated only after a candidate passes this limit, so it
           # records the number of candidates that were actually explored.
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
    """
    Condition A: parent-only backtracking DFS for MiniCrosswords. Mirrors
    _recurse_parent in dfs.py, but works on task.env's board/status
    instead of a text y.
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



# shared beta(c) machinery (crossword-specific)

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
    Crossword version of _beta() in dfs.py. Asks the model for beta(c)
    using the fill history in actions/board_history, from_depth =
    len(actions).

    Call this before popping the failed action off actions/board_history,
    so the model can see it as a possible recovery point.
    """
    from_depth = len(actions)
    if from_depth == 0:
        return 0  # At the root, there is no earlier state to recover to.

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

    # Same cascade detection as _beta() in dfs.py: a cascade means no new
    # candidates were explored since the previous beta(c) call.
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
        print(f"{'  ' * from_depth}[beta(c) BACKTRACK] {reason} — "
              f"depth {from_depth} -> {target_depth}  (skipped {jump - 1} levels){tag}{cascade_tag}")

    return target_depth


# Condition C: deterministic fixed-k=2 backtracking baseline.

def _fixed_k2_crossword(x, actions, board_history, info, to_print, reason='', k=2):
    """
    Crossword analogue of _fixed_k2() in dfs.py: deterministic
    target_depth = max(0, from_depth - k), no LLM call, no cascade concept
    (see _fixed_k2's docstring for the rationale).
    """
    from_depth = len(actions)
    if from_depth == 0:
        return 0  

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
              f"depth {from_depth} -> {target_depth}  (skipped {jump - 1} levels){tag}")

    return target_depth


# Condition B/C: recursive search with configurable non-parent backtracking.
def _diligent_recurse_crossword(env, x, actions, board_history, node_budget, node_snapshots, info,
                                 depth, prune, max_per_state, n_propose, to_print,
                                 backtrack_fn=None):
    """
    Crossword version of _diligent_recurse in dfs.py: every dead end asks
    backtrack_fn where to recover to, instead of always going to the
    parent. There's no early-stop on solved, since crossword search is
    anytime - snapshots are scored by r_word at the end.

    Returns the backtrack target depth. Each frame restores its own env
    state before returning.

    backtrack_fn is the only thing that changes between Condition B
    (_beta_crossword, default) and Condition C (_fixed_k2_crossword).
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
     # Count all ranked candidates before entering the exploration loop. This keeps
    # the evaluation count consistent even when the loop stops early.
    info['candidates_evaluated'] += len(ranked)
    cnt_per_state = 0

    for action in ranked:
        if node_budget[0] <= 0:
            break

        _, _, _, step_info = env.step(action)

        if node_budget[0] > 0 and not any(s == 2 for s in env.status):
            # Cap check happens before incrementing cnt_per_state, so
            # cnt_per_state ends the loop equal to the number of candidates
            # actually explored at this node.
            if cnt_per_state >= max_per_state:
                env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
                break
            cnt_per_state += 1

            node_budget[0] -= 1
            info['nodes_explored'] += 1
            # Track explored nodes separately for cascade detection. This counter records
            # whether any new node has been explored since the previous beta(c)
            # decision, without changing the main node-exploration statistic.
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
                # The requested backtrack target is above the current node, so propagate the
                # request to the caller instead of exploring further siblings.
                info['candidates_pruned'] += len(ranked) - cnt_per_state
                if to_print:
                    print(f"{'  ' * depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
                          f"passed={cnt_per_state} pruned={len(ranked) - cnt_per_state}")
                return bt_depth
            continue   # The backtrack was handled at this level so try the next candidate.

        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)

    info['candidates_pruned'] += len(ranked) - cnt_per_state
    if to_print:
        print(f"{'  ' * depth}[d={depth}] candidates={len(ranked)} evaluated={len(ranked)} "
              f"passed={cnt_per_state} pruned={len(ranked) - cnt_per_state}")
    return backtrack_fn(x, actions, board_history, info, to_print,
                         reason='all candidates exhausted')


def solve_dfs_crossword_nonparent(args, task, idx, to_print=True):
    """Condition B: non-parent backtracking DFS for MiniCrosswords, using beta(c)."""
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

    x = env.render()  # clue/puzzle description shown to the beta(c) model

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
    Condition C: same search as solve_dfs_crossword_nonparent (Condition
    B), but uses _fixed_k2_crossword instead of _beta_crossword.
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


# Condition D: strict non-parent backtracking using beta(c).

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
    Crossword version of _strict_nonparent() in dfs.py. The immediate
    parent is never a legal target once from_depth >= 2. See
    _strict_nonparent in dfs.py for the full logic.
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
        # The model did not provide a valid non-parent target. Use the deepest legal
        # non-parent ancestor as a deterministic fallback, the immediate parent
        # is deliberately excluded by the strict condition.
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
                  f"depth {from_depth} -> {nonparent_fallback_depth}  (parse: {parse_method})")
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
              f"depth {from_depth} -> {target_depth}  (skipped {jump - 1} levels)")
    return target_depth


def solve_dfs_crossword_nonparent_strict(args, task, idx, to_print=True):
    """
    Condition D: same search as Condition B, but uses
    _strict_nonparent_crossword so backtracking never lands on the
    immediate parent.
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