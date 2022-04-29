import numpy as np
import pyspiel
import copy

BOARD_SIZE = 3
game = pyspiel.load_game("hex",{"board_size":BOARD_SIZE})
BLACK, WHITE = 1, -1  # first turn or second turn player

class State:
    '''Board implementation of BOARD_SIZE x BOARD_SIZE Hex Board'''
    X, Y = 'ABCDEFGHI'[0:BOARD_SIZE],  '123456789'[0:BOARD_SIZE]
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE)) # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []
        self.hex_state = game.new_initial_state()

    def __deepcopy__(self):
        newState = State()
        newState.board = copy.deepcopy(self.board)
        newState.win_color = copy.deepcopy(self.win_color)
        newState.record = copy.deepcopy(self.record)
        newState.hex_state = copy.deepcopy(self.hex_state)
        return newState

    def action2str(self, a: int):
        return self.X[a // BOARD_SIZE] + self.Y[a % BOARD_SIZE]

    def str2action(self, s: str):
        return self.X.find(s[0]) * BOARD_SIZE + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        final_bd = [" "+" ".join(self.Y)]
        hex_bd = str(self.hex_state).split("\n")
        for i in range(len(hex_bd)):
            final_bd.append(self.X[i]+" "+hex_bd[i])
        return "\n".join(final_bd)

    def play(self, action):
        # state transition function
        # action is position interger (0~8) or string representation of action sequence
        # Handles the case where action is sequence of actions "0 1 2 3 4"
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        # Single action case
        x, y = action // BOARD_SIZE, action % BOARD_SIZE
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
        # return self.win_color if self.color == BLACK else -self.win_color
        return self.win_color

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(BOARD_SIZE * BOARD_SIZE) if self.board[a // BOARD_SIZE, a % BOARD_SIZE] == 0]

    def feature(self):
        # input tensor for neural net (state)
        # return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)
        observation =  np.array(self.hex_state.observation_tensor(), np.float32)
        return observation.reshape(9,BOARD_SIZE,BOARD_SIZE)[1:BOARD_SIZE+1,:,:]

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        a[0, action // BOARD_SIZE, action % BOARD_SIZE] = 1
        return a