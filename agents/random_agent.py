from core.state import State
from typing import List
import random
import time
from math import sqrt, log

class MCTSAgent:
    def __init__(self) -> None:
        self.best = []
        self.moves = {}
        self.reward = {}

    def think(self, state: State, sim_num: int, temperature:int, show=False) -> None:
        if show:
            print(state)
        start, prev_time = time.time(), 0
        
        if state.terminal():
            return
        
        candidate_moves = state.legal_actions()
        self.moves = {k: 0 for k in candidate_moves}
        self.reward = {k: 0.0 for k in candidate_moves}
        # def ucb_weight(mv, epoch, c=2.0):
        #     expected_reward = self.reward[mv] if mv in self.reward else 0
        #     return expected_reward + c * sqrt(log(epoch)/self.moves[mv])

        for epoch in range(sim_num):
            freshState = state.__copy__()
            first_move = random.choice(list(self.moves))
            freshState.play(first_move)
            not_terminated = not freshState.terminal()

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    pv = self.pv(freshState)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d.'
                          % (tmp_time, state.action2str(pv[0]), self.reward[pv[0]] / self.moves[pv[0]],
                             self.moves[pv[0]], epoch))

            while not_terminated:
                move = random.choice(freshState.legal_actions())
                freshState.play(move)
                not_terminated = not freshState.terminal()
            self.reward[first_move] += freshState.terminal_reward()
                

    def pv(self, state: State) -> List[int]:
        max_value = max(self.reward.values())
        max_moves = [k for k,v in self.reward.items() if v==max_value]
        return random.choice(max_moves)

if __name__=="__main__":
    state = State()

