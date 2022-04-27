import numpy as np
import pyspiel

game = pyspiel.load_game("hex",{"board_size":7})

BLACK, WHITE = 1, -1  # first turn or second turn player

class State:
    '''Board implementation of 7x7 Hex Board'''
    X, Y = 'ABCDEFG',  '1234567'
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((7, 7)) # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []
        self.hex_state = game.new_initial_state()

    def action2str(self, a: int):
        return self.X[a // 7] + self.Y[a % 7]

    def str2action(self, s: str):
        return self.X.find(s[0]) * 7 + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        # output board.
        # s = '   ' + ' '.join(self.Y) + '\n'
        # for i in range(7):
        #     s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(7)]) + '\n'
        # s += 'record = ' + self.record_string()
        # return s
        return str(self.hex_state)

    def play(self, action):
        # state transition function
        # action is position interger (0~8) or string representation of action sequence
        # Handles the case where action is sequence of actions "0 1 2 3 4"
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self


        # Single action case
        x, y = action // 7, action % 7
        self.board[x, y] = self.color
        self.hex_state.apply_action(action)

        # check whether 3 stones are on the line
        if self.hex_state.is_terminal():
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    def terminal(self):
        # terminal state check
        return self.hex_state.is_terminal()

    def terminal_reward(self):
        # terminal reward 
        return self.win_color if self.color == BLACK else -self.win_color

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(7 * 7) if self.board[a // 7, a % 7] == 0]

    def feature(self):
        # input tensor for neural net (state)
        # return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)
        observation =  np.array(self.hex_state.observation_tensor(), np.float32)
        return observation.reshape(9,7,7)[1:8,:,:]

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, 7, 7), dtype=np.float32)
        a[0, action // 7, action % 7] = 1
        return a

state = State().play('B1')
print(state)
print('input feature')
print(state.feature())
state = State().play('B2 A1 C2')
print('input feature')
print(state.feature())