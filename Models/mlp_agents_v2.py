import torch
import torch.nn as nn
import torch.nn.functional as functional

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, num_iters=3):
        super().__init__()

        self.num_iters = num_iters
        self.network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, input_x, answer_y, latent_z):
        input_to_net = torch.cat([latent_z, input_x], dim=1)
        for _ in range(self.num_iters):
            latent_z = self.network(input_to_net)

        aggregate_information = answer_y + latent_z
        input_to_net = torch.cat([aggregate_information, input_x], dim=1)
        answer_y = self.network(input_to_net)

        return answer_y, latent_z
    
class MultiAgentsV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_agents, num_classes=10):
        super().__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.indices_to_feature = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
        self.agents = nn.ModuleList([Agent(input_dim, hidden_dim) for _ in range(num_agents)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_image, target_label=None):
        device = input_image.device
        training_mode = target_label is not None

        target_activation = self.indices_to_feature(target_label).detach() if training_mode else target_label
        answer_y = torch.zeros_like(target_activation, device=device) # Answer
        latent_z = torch.zeros_like(target_activation, device=device) # Agent latent reasoning

        logits_loss = 0.0
        agents_errors = 0.0
        for each in range(self.num_agents):
            # Each agent predict target activation
            agent_prediction, agent_latent_z = self.agents[each](input_image, answer_y, latent_z)

            if training_mode:
                agent_pred_error = functional.mse_loss(agent_prediction, target_activation)
                agents_errors += agent_pred_error

            latent_z = agent_latent_z
        
        logits = self.output_layer(agent_prediction.detach())
        if training_mode: logits_loss = functional.cross_entropy(logits, target_label)
        total_loss = logits_loss + agents_errors

        return logits, total_loss
    
    def infer_each_agent(self, input_image, target_label):
        device = input_image.device

        target_activation = self.indices_to_feature(target_label)
        activation = torch.zeros_like(target_activation, device=device)
        latent_z = torch.zeros_like(target_activation, device=device)

        each_agents_accuracy = torch.zeros(self.num_agents)
        for each in range(self.num_agents):
            # Each agent predict label embedding
            agent_prediction, latent_z = self.agents[each](input_image, activation, latent_z)
            agent_logits = self.output_layer(agent_prediction)
            
            accuracy = (agent_logits.argmax(axis=-1) == target_label).float().mean()
            each_agents_accuracy[each] += accuracy.item()

        return each_agents_accuracy

# model = MultiAgentsV2(input_dim=784, hidden_dim=128, num_agents=3, num_classes=10)
# print(sum(param.numel() for param in model.parameters()))
