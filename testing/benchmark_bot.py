import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir) 
from agents.mcts_agent import MCTSAgent
from core.state import State


agent = MCTSAgent()
state = State()
while True:
  
  distb = agent.think(state, 5000, temperature=1, show=False)
  pv_seq = agent.pv(state)
  state.play(pv_seq[0])

  print(state, state.color)
  if state.terminal():
    break

  while True:
    user_input = input("Input move: ")
    if state.str2action(user_input) in state.legal_actions():
      break
  state.play(user_input)
  if state.terminal():
    break
print(state)
print(state.terminal_reward())