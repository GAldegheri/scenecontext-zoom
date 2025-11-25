import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

rng = np.random.default_rng(123)

def build_trials(n_trials: int, pA: float):
    n_A = int(round(pA * n_trials / 2.0) * 2)  # force even
    n_B = n_trials - n_A
    
    if n_B % 2 == 1:
        n_B -= 1
        n_A += 1
        
    A_trials = [{'cond':'A','type':'same'}]*(n_A//2) + [{'cond':'A','type':'diff'}]*(n_A//2)
    B_trials = [{'cond':'B','type':'same'}]*(n_B//2) + [{'cond':'B','type':'diff'}]*(n_B//2)
    trials = A_trials + B_trials
    random.shuffle(trials)
    return trials


class Staircase2down1up:
    def __init__(self, levels, start_idx=None):
        self.levels = np.array(levels, dtype=float)
        self.idx = start_idx if start_idx is not None else len(levels)//2
        self.correct_streak = 0
   
    def current_delta(self):
        return self.levels[self.idx]
    
    def step(self, correct: bool):
        if correct:
            self.correct_streak += 1
            if self.correct_streak >= 2:
                self.idx = max(0, self.idx - 1)
                self.correct_streak = 0
        else:
            self.idx = min(len(self.levels) - 1, self.idx + 1)
            self.correct_streak = 0
    
def simulate_run(n_trials: int,
                pA: float,
                alpha_A: float = 1.0,
                alpha_B: float = 1.0,
                c_A:float=0.0, c_B:float=0.0,
                update_mode:str='diff_only',
                seed=None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    levels = np.geomspace(0.05, 3.0, 31)
    st = Staircase2down1up(levels)
    trials = build_trials(n_trials, pA)
    rec = {'A':{'H':0,'M':0,'F':0,'CR':0,'acc':0,'n':0},
           'B':{'H':0,'M':0,'F':0,'CR':0,'acc':0,'n':0}}