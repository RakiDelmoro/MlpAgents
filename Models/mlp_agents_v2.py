import torch
import torch.nn as nn
import torch.nn.functional as functional

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, num_iters=6):
        super().__init__()

        self.num_iters = num_iters
        self.network = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, input_x, answer_y, latent_z):
        return self.deep_recursion(input_x, answer_y, latent_z)
    
    def latent_recursion(self, input_x, answer_y, latent_z, n=6):
        input_to_net = torch.cat([latent_z, input_x], dim=1)
        for _ in range(n):
            latent_z = self.network(input_to_net)
        input_to_net = torch.cat([latent_z, input_x], dim=1)
        answer_y = self.network(input_to_net)
        return answer_y, latent_z
    
    def deep_recursion(self, input_emb, answer_y, latent_z, n=6, t=3):
        with torch.no_grad():
            for _ in range(t-1):
                answer_y, latent_z = self.latent_recursion(input_emb, answer_y, latent_z)
        # recursing once to improve answer_y and latent_z
        answer_y, latent_z = self.latent_recursion(input_emb, answer_y, latent_z, n)
        return answer_y, latent_z

class MultiAgentsV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_agents, num_classes=10):
        super().__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.indices_to_feature = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
        self.agents = nn.ModuleList([Agent(input_dim, hidden_dim) for _ in range(num_agents)])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_agents)])

    def forward(self, input_image, target_label=None):
        batch_size = input_image.shape[0]
        device = input_image.device
        training_mode = target_label is not None

        target_activation = self.indices_to_feature(target_label).detach() if training_mode else target_label
        answer_y = torch.zeros(batch_size, self.hidden_dim, device=device) # Answer
        latent_z = torch.zeros(batch_size, self.hidden_dim, device=device) # Agent latent reasoning

        logits_loss = 0.0
        agents_errors = 0.0
        for each in range(self.num_agents):
            # Each agent predict target activation
            agent_prediction, agent_latent_z = self.agents[each](input_image, answer_y, latent_z)
            agent_logits = self.output_layers[each](agent_prediction.detach())

            if training_mode:
                agent_pred_error = functional.mse_loss(agent_prediction, target_activation)
                agent_logits_error = functional.cross_entropy(agent_logits, target_label)

                logits_loss += agent_logits_error
                agents_errors += agent_pred_error

            latent_z = agent_latent_z.detach()

        total_loss = logits_loss + agents_errors

        return agent_logits, total_loss

    def infer_each_agent(self, input_image, target_label):
        device = input_image.device

        target_activation = self.indices_to_feature(target_label)
        activation = torch.zeros_like(target_activation, device=device)
        latent_z = torch.zeros_like(target_activation, device=device)

        each_agents_accuracy = torch.zeros(self.num_agents)
        for each in range(self.num_agents):
            # Each agent predict label embedding
            agent_prediction, latent_z = self.agents[each](input_image, activation, latent_z)
            agent_logits = self.output_layers[each](agent_prediction)
            
            accuracy = (agent_logits.argmax(axis=-1) == target_label).float().mean()
            each_agents_accuracy[each] += accuracy.item()

        return each_agents_accuracy

    def test_each_agent(self, input_image):
        device = input_image.device

        activation = torch.zeros(input_image.shape[0], self.hidden_dim, device=device)
        latent_z = torch.zeros(input_image.shape[0], self.hidden_dim, device=device)

        each_agents_prediction = torch.zeros(self.num_agents)
        for each in range(self.num_agents):
            agent_prediction, latent_z = self.agents[each](input_image, activation, latent_z)
            agent_logits = self.output_layers[each](agent_prediction)
            digit_prediction = agent_logits.argmax(axis=-1)

            each_agents_prediction[each] += digit_prediction.item()

        return each_agents_prediction.type(torch.int64).tolist()

# model = MultiAgentsV2(input_dim=784, hidden_dim=256, num_agents=8, num_classes=10)
# print(sum(param.numel() for param in model.parameters()))
