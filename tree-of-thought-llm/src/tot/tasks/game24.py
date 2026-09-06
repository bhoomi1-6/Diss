import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


_STEP_LINE = re.compile(r'^([-\d./]+)\s*([+\-*/])\s*([-\d./]+)\s*=\s*([-\d./]+)\s*\(left:')


_LEADING_MARKER = re.compile(r'^[-•*]\s+')

_STEP_OPS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: (a / b) if b != 0 else None,
}


def verify_steps(problem: str, output: str):
    """
    Checks the step-by-step trace in the output by redoing each step
    on the original numbers, rather than trusting a separate answer line.

    Returns True or False if a step trace is found, or None if there
    is no trace to check, so the caller can fall back to checking the
    final answer line instead.
    """
    lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
    step_lines = [line for line in lines if '(left:' in line]
    if not step_lines:
        return None

    pool = [sympy.Rational(n) for n in re.findall(r'\d+', problem)]
    for line in step_lines:
        match = _STEP_LINE.match(_LEADING_MARKER.sub('', line))
        if not match:
            return False
        a_str, op, b_str, result_str = match.groups()
        try:
            a, b, result = sympy.Rational(a_str), sympy.Rational(b_str), sympy.Rational(result_str)
        except Exception:
            return False

        remaining = pool.copy()
        if a not in remaining:
            return False
        remaining.remove(a)
        if b not in remaining:
            return False
        remaining.remove(b)

        computed = _STEP_OPS[op](a, b)
        if computed is None or computed != result:
            return False

        pool = remaining + [result]

    return len(pool) == 1 and pool[0] == 24


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file is a csv file
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def is_terminal(self, y: str) -> bool:
        last_line = y.strip().split('\n')[-1]
        return 'left: 24' in last_line

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        verified = verify_steps(self.data[idx], output)
        if verified is not None:
            return {'r': int(verified)}

        # no step trace found, so check the final answer line instead
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value