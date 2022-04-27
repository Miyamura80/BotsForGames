import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir) 
from agents.random_agent import MCTSAgent
from core.state import State


agent = MCTSAgent()
state = State()
while True:
  user_input = input("Input move: ")
  state.play(user_input)
  if state.terminal():
    break
  distb = agent.think(state, 500, temperature=1, show=True)
  pv_seq = agent.pv(state)
  state.play(pv_seq[0])
  print(state)
  if state.terminal():
    break
print(state)
print(state.terminal_reward())