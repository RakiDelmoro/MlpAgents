import torch

class StandardMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = torch.nn.Linear(784, 400, device='cuda')
        self.activation = torch.nn.ReLU()
        self.lin_2 = torch.nn.Linear(400, 10, device='cuda')

    def forward(self, x):
        lin1_out = self.lin_1(x)
        relu_out = self.activation(lin1_out)

        return self.lin_2(relu_out)

# model = StandardMLP()
# print(sum(param.numel() for param in model.parameters()))
