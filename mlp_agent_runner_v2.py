from main import model_runner
from Models.mlp_agents_v2 import MultiAgentsV2

model = MultiAgentsV2(input_dim=784, hidden_dim=128, num_agents=3, num_classes=10)
model_runner(model, use_agents=True)
