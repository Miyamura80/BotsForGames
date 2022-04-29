from typing import List
import random
import time
from math import sqrt, log
from collections import defaultdict
import matplotlib.pyplot as plt

random.seed(1)

class MCTSAgent:
    def __init__(self) -> None:
        self.best = []
        # Both of these :: path -> dict[move, x]
        self.moves = defaultdict(lambda: defaultdict(int))
        self.reward = defaultdict(lambda: defaultdict(float))
    
    def ucb_weight_general(self, state, mv, epoch, c=2.0):
        path = state.record_string()
        expected_reward = self.reward[path][mv]/(self.moves[path][mv]+1)
        n_visit = self.moves[path][mv]
        return expected_reward + c * sqrt(log(epoch)/(n_visit+1))

    def think(self, state: State, sim_num: int, temperature:int, show=False) -> None:
        if show:
            print("Bot to play: \n", state, state.color)
            uncertainties = []

        start, prev_time = time.time(), 0        
        if state.terminal():
            return
        
        init_path = state.record_string()
        for epoch in range(1, sim_num):
            freshState = state.__deepcopy__()
            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    pv = self.pv(freshState)
                    ucb_uncertainty = 2.0 * sqrt(log(epoch)/(self.moves[init_path][pv[0]]+1))
                    uncertainties.append(ucb_uncertainty)
                    print(f"Uncertainty: {ucb_uncertainty}")
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d.'
                          % (tmp_time, state.action2str(pv[0]), self.reward[init_path][pv[0]] / (self.moves[init_path][pv[0]]+1), 
                            self.moves[init_path][pv[0]], epoch))
            not_terminated = True
            rewards = []
            while not_terminated:
                # first_move = random.choice(list(self.moves))
                path = freshState.record_string()
                ucb_weights = [self.ucb_weight_general(freshState, k, epoch) for k in freshState.legal_actions()]
                max_ucb_weight = max(ucb_weights)
                move = [k for k in freshState.legal_actions() if self.ucb_weight_general(freshState, k, epoch)==max_ucb_weight][0]
                if move in self.moves[path]:
                  self.moves[path][move] += 1
                else:
                  self.moves[path][move] = 1
                freshState.play(move)
                if path not in self.reward:
                  self.reward[path] = {move: 0}
                rewards.append((self.reward[path], move))  
                not_terminated = not freshState.terminal()
            for (r,m) in rewards:
                r[m] += freshState.terminal_reward()
        if show:
            plt.plot(uncertainties)
            plt.show()

    def pv(self, state: State) -> List[int]:
        path = state.record_string()
        if path in self.reward:
          max_value = max(self.reward[path].values())
          max_moves = [k for k,v in self.reward[path].items() if v==max_value]
          print(f"Max Value: {max_value} Rewards: {self.reward[path]} Moves: {self.moves[path]}")
        else:
          max_moves = state.legal_actions()
          print("ah")
        return [random.choice(max_moves)]
