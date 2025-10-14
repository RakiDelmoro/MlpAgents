import torch
import torch.nn as nn

def squash(tensor, axis=-1):
    """
    Squash function: makes vector length between 0 and 1
    - Short vectors -> nearly zero
    - Long vectors -> nearly 1 (but keeps direction)
    """
    squared_norm = torch.sum(torch.square(tensor), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * tensor

class AgentMarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, agents_probability, targets):
        present_loss = torch.nn.functional.relu(self.m_plus - agents_probability) ** 2
        absent_loss = torch.nn.functional.relu(agents_probability - self.m_minus) ** 2
        loss = targets * present_loss + self.lambda_ * (1 - targets) * absent_loss
        total_loss = loss.sum(dim=-1).mean()

        return total_loss

class Agent(nn.Module):
    def __init__(self, input_dim_size=784, out_dim_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_dim_size, out_dim_size, device='cuda')
        self.activation_fn = nn.ReLU()
        self.linear2 = nn.Linear(out_dim_size, out_dim_size, device='cuda')

    def forward(self, x):
        return squash(self.linear2(self.activation_fn(self.linear1(x))))

class MultiAgents(nn.Module):
    def __init__(self, num_in_agents=4, num_out_agents=10, in_agents_dim_size=256, out_agents_dim_size=32):
        super().__init__()

        self.num_in_agents = num_in_agents
        self.num_out_agents = num_out_agents
        self.in_agents_dim_size = in_agents_dim_size
        self.out_agents_dim_size = out_agents_dim_size

        self.first_agents = nn.ModuleList([Agent() for _ in range(num_in_agents)])
        self.W = nn.Parameter(torch.randn(1, num_in_agents, num_out_agents, out_agents_dim_size, in_agents_dim_size, device='cuda'))

    def forward(self, x):
        first_agent_outputs = []
        for each in self.first_agents:
            output = each(x)
            first_agent_outputs.append(output)

        stack_first_agents_output = torch.stack(first_agent_outputs, dim=1).unsqueeze(2).unsqueeze(-1)
        self.w_batch = self.W.repeat(x.shape[0], 1, 1, 1, 1)
        each_agent_pred = torch.matmul(self.w_batch, stack_first_agents_output).squeeze(-1)

        logits_for_routing = torch.zeros((x.shape[0], self.num_in_agents, self.num_out_agents, 1), device='cuda')
        for iteration in range(2):
            probabilities_to_route = torch.nn.functional.softmax(logits_for_routing, dim=2)
            second_agents_logit_outputs = (probabilities_to_route * each_agent_pred).sum(dim=1, keepdim=True)
            second_agents_outputs = squash(second_agents_logit_outputs, axis=-1)

            if iteration < 3 - 1:  # Don't update on last iteration
                agreement = (each_agent_pred * second_agents_outputs).sum(dim=-1, keepdim=True)
                logits_for_routing += agreement

        return torch.linalg.norm(second_agents_outputs.squeeze(1), dim=-1)

