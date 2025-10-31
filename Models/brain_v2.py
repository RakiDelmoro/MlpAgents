import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, input_size, output_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes

        self.encode_label = self.label_to_bits()
        self.connection = torch.zeros(input_size, output_size, device='cuda')

    def label_to_bits(self, sparsity=0.03):
        bits = torch.zeros(self.num_classes, self.output_size, device='cuda')
        top_k = int(sparsity * self.output_size)

        for each in range(self.num_classes):
            indices = torch.randperm(self.output_size)[:top_k]
            bits[each, indices] = 1
        return bits

    def encode_input(self, x, sparsity_pct=3):
        binary_input = (x > 0).float()
        top_k = max(1, int(x.shape[-1] * sparsity_pct / 100))
        indices = torch.topk(x.abs(), k=top_k, dim=-1, largest=True)[1]
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)

        return binary_input * mask
    
    def update(self, image_bits, label_bits):
        image_active_bits = torch.where(image_bits > 0)[1]
        label_active_bits = torch.where(label_bits > 0)[1]

        # If both firing make connection
        self.connection[image_active_bits[:, None], label_active_bits[None, :]] += 0.1
        self.connection.clamp(0, 1)

    def forward(self, image, label, learn=True):
        image_as_bits = self.encode_input(image)
        label_as_bits = self.encode_label[label]

        if learn: self.update(image_as_bits, label_as_bits)

    def predict(self, image, label):
        image_as_bits = self.encode_input(image)
        label_as_bits = self.encode_label[label]
        predicted = torch.matmul(image_as_bits, self.connection)

        print(label_as_bits == predicted)     

x = torch.randn(1, 784, device='cuda')
y = torch.randint(0, 10, size=(1,), device='cuda')
model = Agent(784, 1024, 10)

# Training
model(x, y)
# Prediction
model.predict(x, y)
