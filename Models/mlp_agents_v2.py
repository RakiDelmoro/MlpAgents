import torch
import torch.nn as nn
import torch.nn.functional as functional

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, num_iters=6):
        super().__init__()

        self.num_iters = num_iters
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, input_x):
        return self.mlp_layers(input_x)

class MultiAgentsV2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_agents, num_classes=10):
        super().__init__()

        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.indices_to_feature = [torch.randn(num_classes, hidden_dim, device='cuda') for _ in range(num_agents)]
        self.agents = nn.ModuleList([Agent(input_dim, hidden_dim) for _ in range(num_agents)])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def latent_recursion(self, agent, input_x, answer_y, latent_z, n=6):
        input_to_net = torch.cat([latent_z, input_x], dim=-1)
        for _ in range(n):
            latent_z = agent(input_to_net)
        input_to_net = torch.cat([latent_z, input_x], dim=-1)
        answer_y = agent(input_to_net)
        return answer_y, latent_z

    def deep_recursion(self, agent, input_emb, answer_y, latent_z, n=6, t=3):
        with torch.no_grad():
            for _ in range(t-1):
                answer_y, latent_z = self.latent_recursion(agent, input_emb, answer_y, latent_z)
        # recursing once to improve answer_y and latent_z
        answer_y, latent_z = self.latent_recursion(agent, input_emb, answer_y, latent_z, n)
        return answer_y, latent_z
    
    def forward(self, input_image, target_label=None):
        batch_size = input_image.shape[0]
        device = input_image.device
        training_mode = target_label is not None
    
        answer_y = torch.zeros(batch_size, self.hidden_dim, device=device) # Answer
        latent_z = torch.zeros(batch_size, self.hidden_dim, device=device) # Agent latent reasoning

        agents_errors = 0.0
        for each in range(self.num_agents):
            agent_prediction, agent_latent_z = self.deep_recursion(self.agents[each], input_image, answer_y, latent_z)
            one_hot_encoded = self.decode_agent_prediction(agent_prediction, self.indices_to_feature[each])
            if training_mode:
                agent_target_activation =  self.indices_to_feature[each][target_label]
                agent_pred_error = functional.mse_loss(agent_prediction, agent_target_activation)

                agents_errors += agent_pred_error

                self.optimizer.zero_grad()
                agent_pred_error.backward()
                self.optimizer.step()

            latent_z = agent_latent_z.detach()

        total_loss = agents_errors

        return one_hot_encoded, total_loss

    def decode_agent_prediction(self, predicted_activation, target_activation):
        distances = torch.cdist(predicted_activation, target_activation)
        probabilities = 1 - distances.softmax(dim=-1)
        return probabilities

    def infer_each_agent(self, input_image, target_label):
        batch_size = input_image.shape[0]
        device = input_image.device

        activation = torch.zeros(batch_size, self.hidden_dim, device=device)
        latent_z = torch.zeros(batch_size, self.hidden_dim, device=device)

        each_agents_accuracy = torch.zeros(self.num_agents)
        for each in range(self.num_agents):
            agent_prediction, latent_z = self.deep_recursion(self.agents[each], input_image, activation, latent_z)
            one_hot_encoded = self.decode_agent_prediction(agent_prediction, self.indices_to_feature[each])
            
            accuracy = (one_hot_encoded.argmax(axis=-1) == target_label).float().mean()
            each_agents_accuracy[each] += accuracy.item()

        return each_agents_accuracy

    def test_each_agent(self, input_image):
        batch_size = input_image.shape[0]
        device = input_image.device

        activation = torch.zeros(batch_size, self.hidden_dim, device=device)
        latent_z = torch.zeros(batch_size, self.hidden_dim, device=device)
    
        each_agents_prediction = torch.zeros(self.num_agents)
        agents_one_hot_encoded = []
        for each in range(self.num_agents):
            agent_prediction, latent_z = self.deep_recursion(self.agents[each], input_image, activation, latent_z)
            one_hot_encoded = self.decode_agent_prediction(agent_prediction, self.indices_to_feature[each])
            digit_prediction = one_hot_encoded.argmax(axis=-1)

            each_agents_prediction[each] += digit_prediction.item()
            agents_one_hot_encoded.append(one_hot_encoded)

        return each_agents_prediction.type(torch.int64).tolist(), agents_one_hot_encoded

# model = MultiAgentsV2(input_dim=784, hidden_dim=256, num_agents=8, num_classes=10)
# print(sum(param.numel() for param in model.parameters()))
