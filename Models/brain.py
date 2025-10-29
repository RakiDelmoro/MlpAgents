import torch
import torch.nn as nn
from collections import Counter

class Agent(nn.Module):
    def __init__(self, input_size, receptive_field_size, num_neurons: int, local_neuron_act_sparsity: float = 0.01,
                 synapse_threshold: float = 0.5, boost_strength: float = 0.01):
        super().__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.sparsity = local_neuron_act_sparsity
        self.synapse_threshold = synapse_threshold
        self.boost_strength = boost_strength
        
        self.input_h, self.input_w = input_size[0], input_size[1]
        self.spatial_h, self.spatial_w = receptive_field_size[0], receptive_field_size[1]

        # Build and store receptive field mask
        self.receptive_field_connection = self._build_receptive_field_connection()

        # Permanent synaptic connections (0-1), initialized with receptive field
        permanences_init = torch.rand(num_neurons, input_size[0]*input_size[1], device='cuda') * 0.7
        permanences_init = permanences_init * self.receptive_field_connection
        self.permanences = nn.Parameter(permanences_init, requires_grad=False)

        self.boost = torch.ones(num_neurons, device='cuda')
        self.neuron_active_cycle = torch.zeros(num_neurons, device='cuda')
        self.neuron_num_connectivity_cycle = torch.zeros(num_neurons, device='cuda')

    def encode_input(self, x):
        """Convert input to binary using threshold."""
        return (x > 0).float()

    def neurons_compete_within_neighborhood(self, activations):
        patches_h = self.input_h // self.spatial_h
        patches_w = self.input_w // self.spatial_w
        total_patches = patches_h * patches_w
        patch_ids = torch.arange(self.num_neurons, device=activations.device) % total_patches

        active = torch.zeros_like(activations)
        for patch_idx in range(total_patches):
            mask = patch_ids == patch_idx
            k = max(1, int(mask.sum() * self.sparsity))
            topk_vals, topk_idx = torch.topk(activations[:, mask], k=k, dim=1)
            patch_acts = torch.zeros_like(activations[:, mask])
            patch_acts.scatter_(1, topk_idx, 1.0)
            active[:, mask] = patch_acts

        return active, torch.where(active > 0)[1]

    def _build_receptive_field_connection(self):
        patches_h = self.input_h // self.spatial_h
        patches_w = self.input_w // self.spatial_w
        total_patches = patches_h * patches_w

        spatial_connection = torch.zeros(self.num_neurons, self.input_h*self.input_w, device='cuda')
        for neuron_idx in range(self.num_neurons):
            # Map column to a patch (cycles through patches if columns > patches)
            patch_idx = neuron_idx % total_patches
            # Convert patch index to 2D coordinates
            patch_row = patch_idx // patches_w
            patch_col = patch_idx % patches_w

            # Calculate patch bounds in input space
            h_min = patch_row * self.spatial_h
            h_max = h_min + self.spatial_h
            w_min = patch_col * self.spatial_w
            w_max = w_min + self.spatial_w
            
            # Set mask for this column's patch
            for h in range(h_min, h_max):
                for w in range(w_min, w_max):
                    input_idx = h * self.input_w + w
                    spatial_connection[neuron_idx, input_idx] = 1.0

        return spatial_connection
    
    def compute_num_connected(self, input_activation):
        """Compute connected scores between input and neuron synapses."""
        # Create binary connectivity from permanences
        connected = (self.permanences > self.synapse_threshold).float()
        # Overlap = number of connected synapses that are active
        overlap = torch.matmul(input_activation, connected.t())
        return overlap

    def active_neuron_cycles(self, active, neurons_activation):
        """Update duty cycles based on raw overlap (before boosting)."""
        decay = 0.999
        active_mean = active.mean(dim=0)
        self.neuron_active_cycle.mul_(decay).add_(active_mean, alpha=1-decay)

    def _update_boost(self):
        target_duty_cycle = self.sparsity
        min_duty_cycle = 0.001
        
        active_dc = torch.clamp(self.neuron_active_cycle, min=min_duty_cycle)
        boost = target_duty_cycle / active_dc
    
        self.boost = 1.0 + (boost - 1.0) * self.boost_strength
        self.boost = torch.clamp(self.boost, min=1.0, max=10.0)

    def _learn(self, input_activation, neurons_activation):
        """Update permanences using Hebbian learning (vectorized)."""
        permanences_inc = 0.05
        permanences_dec = 0.015

        # Determine which neurons were active in this batch
        neurons_active_mask = neurons_activation.sum(dim=0) > 0  # [num_neurons]
        if neurons_active_mask.sum() == 0:
            return  # skip if no active neurons

        # Use binary input averaged across batch
        x_mean = input_activation.mean(dim=0, keepdim=True)  # [1, num_inputs]

        # Restrict to active neurons and their receptive fields
        rf_mask = self.receptive_field_connection[neurons_active_mask]  # [N_active, num_inputs]
        p = self.permanences.data[neurons_active_mask]

        # Hebbian update: increase for active inputs, decrease for inactive
        delta = (x_mean * permanences_inc) - ((1 - x_mean) * permanences_dec)
        p += delta * rf_mask

        # Clamp and reapply receptive field mask
        p.clamp_(0, 1)
        p *= rf_mask

        # Store back to global permanences
        self.permanences.data[neurons_active_mask] = p

    def forward(self, x: torch.Tensor, learn: bool = True):
        """Forward pass with spatial pooling."""
        input_binary = self.encode_input(x)
    
        neurons_num_connected = self.compute_num_connected(input_binary)
        boosted_neurons_num_connected = neurons_num_connected * self.boost.unsqueeze(0)
        neurons_activation, neurons_activated_indices = self.neurons_compete_within_neighborhood(boosted_neurons_num_connected)

        self.active_neuron_cycles(neurons_activation, neurons_num_connected)
        self._update_boost()

        if learn:
            self._learn(input_binary, neurons_activated_indices)
        
        return neurons_activation, neurons_activated_indices

    def predict(self, new_activation, model_memories):
        '''Use cosine similarity'''
        new_activation = new_activation.flatten().tolist()
        matches = []
        for memory in model_memories:
            memory_activation = torch.zeros(self.num_neurons)
            activation = torch.zeros(self.num_neurons)
            memory_activation[memory['neurons_activation']] = 1
            activation[new_activation] = 1

            theta = int(0.7 * memory_activation.sum())
            overlap = torch.dot(memory_activation, activation)
            match = overlap >= theta

            if match:
                matches.append((overlap, memory['label']))

        if matches == []:
            return 0, 0

        top_overlap = sorted(matches, reverse=True)[:5]
        top_scores_labels = [label for _, label in top_overlap]

        votes = Counter(top_scores_labels)
        return votes.most_common(1)[0][0], top_overlap[0][0]

    def phase_1(self, train_loader):
        '''Phase 1 is equavalent to learning. interacting to the task we want to learn'''
        print('Learning...')
        for image, _ in train_loader:
            image = torch.tensor(image, device='cuda')
            _, _ = self.forward(image, learn=True)
        print('Done learning!')

    def phase_2(self, memory_loader):
        '''Phase 2 is equavalent to memory. assigning label to the learned sparse neurons activation after phase 1'''
        model_memory = []
        print('Adding memory....')
        for image, label in memory_loader:
            image = torch.tensor(image, device='cuda')

            _, active_neurons_indices = self.forward(image, learn=False)
            model_memory.append({'neurons_activation': active_neurons_indices.flatten().tolist(),
                                 'label': str(label.item())})
        print('Done!')

        return model_memory

    def test(self, model_memory, test_loader):
        correctness = []
        scores = []
        for image, label in test_loader:
            batched_image = torch.tensor(image, device='cuda')

            _, active_neurons_indices = self.forward(batched_image, learn=False)
            predicted, score = self.predict(active_neurons_indices, model_memory)

            correct = int(int(predicted) == label.item())
            scores.append(score)
            correctness.append(correct)

        return sum(correctness) / len(correctness), sum(scores) / len(scores)
