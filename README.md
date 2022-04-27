# BotsForGames
For 3rd Year Computer Science project "Bots for games" at University of Oxford under Stefan Kiefer

Load the model from pickle as such:
```
model = Net()
model.load_state_dict(torch.load('network.pkl'))
model.eval()
```
