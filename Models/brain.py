import torch
import torch.nn as nn
import numpy as np

class DigitMemory:
    def __init__(self, num_neurons, device='cuda'):
        self.num_neurons = num_neurons
        self.device = device
        self.memory = torch.zeros(num_neurons, dtype=torch.bool, device=device)
        self.num_vectors = 0

    def add(self, feature):
        """Add a new feature to the memory"""
        self.memory = torch.logical_or(self.memory, feature)
        self.num_vectors += 1

    def reset(self):
        """Clear the union for a new training epoch"""
        self.memory = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        self.num_vectors = 0

    def match_score(self, new_feature):
        """Calculate overlap score between memory and new feature"""
        overlap = torch.logical_and(new_feature, self.memory).sum().float()
        new_feature_active = new_feature.sum().float()
        memory_active = self.memory.sum().float()
        
        if new_feature_active == 0:
            return torch.tensor(0.0, device=self.device)
        memory_size = (new_feature_active + memory_active - overlap)
        return overlap / memory_size
    
    def get_active_neurons(self):
        """Return number of ON bits in union"""
        return self.memory.sum().item()

class Agent(nn.Module):
    def __init__(self, input_size: int, num_neurons: int, sparsity: float = 0.02,
                 synapse_threshold: float = 0.5, boost_strength: float = 1.0, potential_connection=0.2, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.num_agents = num_neurons
        self.sparsity = sparsity
        self.synapse_threshold = synapse_threshold
        self.boost_strength = boost_strength
        
        self.receptive_field_radius = 8
        self.class_unions = [DigitMemory(num_neurons) for _ in range(num_classes)]
        self.potential_connection_mask = self._build_receptive_field_connection(potential_connection)

        # Permanent synaptic connections (0-1)
        self.permanences = nn.Parameter((torch.rand(num_neurons, input_size, device='cuda') * 0.4 + 0.3) * self.potential_connection_mask.float(), requires_grad=False)
        
        # Boosting factors for homeostatic regulation
        self.register_buffer('boost_factors', torch.ones(num_neurons, device='cuda'))
        self.register_buffer('active_duty_cycle', torch.zeros(num_neurons, device='cuda'))

    def binary_input(self, x):
        return (x > 0.5).float()

    def compute_overlap(self, x: torch.Tensor) -> torch.Tensor:
        """Compute overlap scores between input and neuron synapses."""
        connected = (self.permanences > self.synapse_threshold).float()
        num_connected = connected.sum(dim=1, keepdim=True).clamp_min(1.0)
        overlap = torch.matmul(x, connected.t()) / num_connected.t()
        return overlap * self.boost_factors.unsqueeze(0)

    def forward(self, x: torch.Tensor, learn: bool = True):
        normalized_x = self.binary_input(x)
        overlap = self.compute_overlap(normalized_x)

        # Winner-take-all with sparsity constraint
        k = max(1, int(self.num_agents * self.sparsity))
        topk_values, top_neuron_indices = torch.topk(overlap, k, dim=-1)

        active_neurons = torch.zeros_like(overlap, device='cuda')
        active_neurons.scatter_(1, top_neuron_indices, 1.0)

        if learn:
            self._learn(x, active_neurons)

        return active_neurons, top_neuron_indices
    
    def _build_receptive_field_connection(self, potential_connection):
        H, W = 28, 28
        patch_h, patch_w = 16, 16
        mask = torch.zeros(self.num_agents, self.input_size, device='cuda')

        for ci in range(patch_h):
            for cj in range(patch_w):
                col_idx = ci * patch_w + cj

                # Map column center to input coordinates
                center_y = int((ci + 0.5) * H / H)
                center_x = int((cj + 0.5) * W / W)

                # Define receptive field window
                for yi in range(max(0, center_y - self.receptive_field_radius),
                                min(H, center_y + self.receptive_field_radius)):
                    for xi in range(max(0, center_x - self.receptive_field_radius),
                                    min(W, center_x + self.receptive_field_radius)):
                        if torch.rand(1).item() < potential_connection:
                            mask[col_idx, yi * W + xi] = 1.0
        return mask

    def _learn(self, x: torch.Tensor, agents: torch.Tensor):
        """Update permanences using Hebbian learning."""
        self.active_duty_cycle *= 0.999
        self.active_duty_cycle += 0.001 * agents.mean(dim=0)

        for i in range(x.shape[0]):
            active_agents = agents[i].nonzero(as_tuple=True)[0]
            if len(active_agents) > 0:
                self.permanences[active_agents] += 0.05 * x[i].unsqueeze(0)
                self.permanences[active_agents] -= 0.008 * (1 - x[i].unsqueeze(0))

        self.permanences.data.clamp_(0, 1)
        
        target_density = self.sparsity
        self.boost_factors = torch.exp(
            self.boost_strength * (target_density - self.active_duty_cycle))

    def reset_memory(self):
        for union in self.class_unions:
            union.reset()
        
    def add_to_memory(self, feature, label):
        self.class_unions[label].add(feature)

    def predict(self, feature):
        scores = torch.tensor([
            union.match_score(feature).item() 
            for union in self.class_unions], device='cuda')

        predicted_label = torch.argmax(scores).item()
        return predicted_label, scores
    
    def check_stability(self, test_loader, previous_features, num_samples=1000):
        current_features = []
        with torch.no_grad():
            for image, _ in test_loader:
                batched_image = torch.tensor(image, requires_grad=True, device='cuda')
                features, _ = self.forward(batched_image, learn=False)
                for i in range(batched_image.shape[0]):
                    current_features.append(features[i].clone())

                if len(current_features) >= num_samples:
                    break

        current_features = current_features[:num_samples]
        print("Mean sparsity:", torch.stack(current_features).mean().item())

        if previous_features is None:
            return None, current_features

        stabilities = []
        for currrent, previous in zip(current_features, previous_features):
            overlap = (currrent * previous).sum()
            total = currrent.sum()
            if total > 0:
                stabilities.append((overlap / total).item())
        stability = sum(stabilities) / len(stabilities) if stabilities else 0.0

        return stability, current_features

    def phase_1(self, train_loader, test_loader, epoch, previous_features):
        '''Phase 1 is equavalent to learning. interacting to the task we want to learn'''

        for batch_idx, (image, label) in enumerate(train_loader):
            batched_image = torch.tensor(image, requires_grad=True, device='cuda')
            _, _ = self.forward(batched_image, learn=True)

        stability, current_features = self.check_stability(test_loader, previous_features=previous_features)
        if stability is None:
            print(f"Epoch {epoch} - First epoch, baseline set")
        else:
            print(f"Epoch {epoch} - Features Overlap with prev epoch: {stability:.3f}")
            print(f"  (1.0 = no change, <0.9 = still learning)")
        
        # Check if learning has plateaued
        if stability is not None and stability > 0.95:
            print(f"Representations stabilized! Stopping early at epoch {epoch}")
        
        return current_features

    def phase_2(self, train_loader, test_loader):
        '''Phase 2 is equavalent to memory. assigning label to the features after phase 1'''

        with torch.no_grad():
            print('Creating memory...')
            for image, label in train_loader:
                batched_image = torch.tensor(image, requires_grad=True, device='cuda')
                batched_label = torch.tensor(label, device='cuda')

                features, _ = self.forward(batched_image, learn=False)
                self.add_to_memory(features, batched_label)
            print('Done creating memory')

            correctness = []
            for image, label in test_loader:
                batched_image = torch.tensor(image, device='cuda')
                batched_label = torch.tensor(label, device='cuda')

                features, _ = self.forward(batched_image, learn=False)
                predicted, scores = self.predict(features)

                correct = int(predicted == batched_label.item())
                correctness.append(correct)

            print(f'Accuracy: {sum(correctness) / len(correctness)}')

        torch.save(self.forward, 'thousand_agents.pth')

# input_x = torch.rand(1, 784, device='cuda')
# model = SpatialPooler(input_size=784, num_agents=2048, sparsity=0.02)
# print(model(input_x).shape)
