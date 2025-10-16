import torch
import torch.nn.functional as functional

class StandardMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = torch.nn.Linear(784, 2000, device='cuda')
        self.activation = torch.nn.ReLU()
        self.lin_2 = torch.nn.Linear(2000, 10, device='cuda')

    def forward(self, input_image, target_label=None):
        loss = 0.0
        lin1_out = self.lin_1(input_image)
        relu_out = self.activation(lin1_out)
        logits = self.lin_2(relu_out)

        if target_label is not None:
            loss = functional.cross_entropy(logits, target_label)
        return logits, loss

# model = StandardMLP()
# print(sum(param.numel() for param in model.parameters()))
