import torch
import torch.nn as nn
import torch.nn.functional as functional

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, input_x, previous_activation):
        input_to_net = torch.cat([previous_activation, input_x], dim=1)
        return self.network(input_to_net)

class MultiAgentsV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_agents, num_classes=10):
        super().__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.indices_to_feature = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
        self.agents = nn.ModuleList([Agent(input_dim, hidden_dim) for _ in range(num_agents)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_image, target_label=None):
        batch_size = input_image.shape[0]
        device = input_image.device
        training_mode = target_label is not None

        target_activation = self.indices_to_feature(target_label) if training_mode else target_label
        activation = torch.randn(batch_size, self.hidden_dim, device=device)

        agents_out = []
        logits_loss = 0.0
        agents_errors = 0.0
        for each in range(self.num_agents):
            # Each agent predict target activation
            agent_prediction = self.agents[each](input_image, activation)

            if training_mode:
                agent_pred_error = functional.mse_loss(agent_prediction, target_activation)
                agents_errors += agent_pred_error

            activation = agent_prediction
            agents_out.append(agent_prediction)

        logits = self.output_layer(activation.detach())
        if training_mode: logits_loss = functional.cross_entropy(logits, target_label)
        total_loss = logits_loss + agents_errors

        return logits, total_loss

    def infer_each_agent(self, input_image, target_label):
        batch_size = input_image.shape[0]
        device = input_image.device

        activation = torch.randn(batch_size, self.hidden_dim, device=device)
        agents_outs = []
        for each in range(self.num_agents):
            # Each agent predict label embedding
            embed_pred = self.agents[each](input_image, activation)
            activation = embed_pred
            agents_outs.append(embed_pred)

        each_agents_accuracy = torch.zeros(self.num_agents)
        for i, out in enumerate(agents_outs):
            agent_logits = self.output_layer(out)
            batch_accuracy = (agent_logits.argmax(axis=-1) == target_label).float().mean()

            each_agents_accuracy[i] += batch_accuracy.item()

        return each_agents_accuracy

# model = MultiAgentsV2(input_dim=784, hidden_dim=128, num_agents=3, num_classes=10)
# print(sum(param.numel() for param in model.parameters()))
