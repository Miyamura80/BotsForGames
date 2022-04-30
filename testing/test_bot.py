import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir) 
from agents.mcts_agent import MCTSAgent
from core.state import State


state = State()

def benchmark_bots(agent0, agent1, num=20, show=False):
  wins = [0,0]
  for epoch in range(num):
    
    distb = agent0.think(state, 5000, temperature=1, show=False)
    pv_seq = agent0.pv(state)
    state.play(pv_seq[0])

    if show:
      print(state, state.color)
    if state.terminal():
      wins[0] += 1
      continue
      
    distb = agent1.think(state, 5000, temperature=1, show=False)
    pv_seq = agent1.pv(state)
    state.play(pv_seq[0])
  
    if show:
      print(state, state.color)
  
    if state.terminal():
      wins[1] += 1
      continue

  print(state)
  print(state.terminal_reward())
  

if __name__=="__main__":
  agent1 = MCTSAgent()
  agent2 = MCTSAgent()
  print(benchmark_bots(agent1, agent2))
