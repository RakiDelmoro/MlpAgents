import torch
import math
import torch.nn as nn
from collections import Counter

class Agent(nn.Module):
    def __init__(self, input_size, receptive_field_size, num_neurons: int, local_neuron_act_sparsity: float = 0.05,
                 synapse_threshold: float = 0.5, boost_strength: float = 0.005):
        super().__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.sparsity = local_neuron_act_sparsity
        self.synapse_threshold = synapse_threshold
        self.boost_strength = boost_strength
        
        self.input_h, self.input_w = input_size[0], input_size[1]
        self.spatial_h, self.spatial_w = receptive_field_size[0], receptive_field_size[1]
        self.total_patches = (self.input_h // self.spatial_h) * (self.input_w // self.spatial_w)

        # Build and store receptive field mask
        self.receptive_field_connection = self._build_receptive_field_connection()
        self.neuron_patch_indices = self._precompute_neuron_patch_indices()

        # Permanent synaptic connections (0-1), initialized with receptive field
        permanences_init = torch.rand(num_neurons, input_size[0]*input_size[1], device='cuda') * 0.7
        permanences_init = permanences_init * self.receptive_field_connection
        self.permanences = nn.Parameter(permanences_init, requires_grad=False)

        self.boost = torch.ones(num_neurons, device='cuda')
        self.neuron_active_cycle = torch.zeros(num_neurons, device='cuda')
        self.neuron_num_connectivity_cycle = torch.zeros(num_neurons, device='cuda')

        self.target_duty_cycle = self.sparsity
        self.min_duty_cycle = 0.001

    def encode_input(self, x, sparsity_pct=5):
        binary_input = (x > 0).float()
        top_k = max(1, int(x.shape[-1] * sparsity_pct / 100))
        indices = torch.topk(x.abs(), k=top_k, dim=-1, largest=True)[1]
        mask = torch.zeros_like(x)
        mask.scatter_(-1, indices, 1.0)

        return binary_input * mask

    def neurons_compete(self, overlap):
        batch_size = overlap.shape[0]
        activation = torch.zeros_like(overlap)
        
        for b in range(batch_size):
            overlap_b = overlap[b]  # [num_neurons]
            
            # Compete within each patch
            for patch_idx in range(self.total_patches):
                # Get all neurons with this patch receptive field
                patch_mask = self.neuron_patch_indices == patch_idx
                patch_neurons = torch.where(patch_mask)[0]

                if len(patch_neurons) == 0: continue
                
                # Get raw overlap for this patch
                patch_overlap_raw = overlap_b[patch_neurons]
                
                # K-winners within this patch
                k = max(1, int(len(patch_neurons) * self.sparsity))
                k = min(k, len(patch_neurons))
                
                # RAW K-winners (for duty cycle tracking)
                topk_vals_raw, topk_indices_raw = torch.topk(patch_overlap_raw, k=k, largest=True)
                winners_mask_raw = topk_vals_raw > 0
                winners_raw = patch_neurons[topk_indices_raw[winners_mask_raw]]
                activation[b, winners_raw] = 1.0
                activation[b, winners_raw] = 1.0
        
        return activation, torch.where(activation > 0)[1]

    def _precompute_neuron_patch_indices(self):        
        # Direct mapping: each neuron gets its patch index
        self._neuron_patch_idx = torch.zeros(self.num_neurons, dtype=torch.long, device='cuda')
        for neuron_idx in range(self.num_neurons): self._neuron_patch_idx[neuron_idx] = neuron_idx % self.total_patches
        return self._neuron_patch_idx

    def _build_receptive_field_connection(self):
        """Build spatial receptive field connections efficiently.
        Vectorized: compute each patch's receptive field once, assign to all neurons in that patch.
        """
        patches_w = self.input_w // self.spatial_w

        spatial_connection = torch.zeros(self.num_neurons, self.input_h * self.input_w, device='cuda')
        # Compute each patch's receptive field once
        patch_rf_cache = {}
        for patch_idx in range(self.total_patches):
            patch_row = patch_idx // patches_w
            patch_col = patch_idx % patches_w

            h_min = patch_row * self.spatial_h
            h_max = h_min + self.spatial_h
            w_min = patch_col * self.spatial_w
            w_max = w_min + self.spatial_w

            # Vectorized index computation
            h_indices = torch.arange(h_min, h_max, device='cuda')
            w_indices = torch.arange(w_min, w_max, device='cuda')
            h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
            input_indices = h_grid * self.input_w + w_grid
            
            patch_rf_cache[patch_idx] = input_indices.flatten()
        
        # Assign each neuron to its patch's receptive field
        for neuron_idx in range(self.num_neurons):
            patch_idx = neuron_idx % self.total_patches
            spatial_connection[neuron_idx, patch_rf_cache[patch_idx]] = 1.0

        return spatial_connection
    
    def compute_num_connected(self, input_activation):
        """Compute connected scores between input and neuron synapses."""
        # Create binary connectivity from permanences
        connected = (self.permanences > self.synapse_threshold).float()
        # Overlap = number of connected synapses that are active
        overlap = torch.matmul(input_activation, connected.t())
        return overlap

    def _learn(self, input_activation, neurons_activation):
        permanences_inc = 0.05
        permanences_dec = 0.015

        # Determine which neurons were active in this batch
        neurons_active_mask = neurons_activation.sum(dim=0) > 0
        if neurons_active_mask.sum() == 0: return
        x_mean = input_activation.mean(dim=0, keepdim=True)
        n_mean = neurons_activation.mean(dim=0)
        # strengthen when BOTH fire, weaken when they don't correlate
        correlation = x_mean * n_mean[neurons_active_mask].unsqueeze(1)
        # Increase when correlated, decrease when not
        update = (correlation * permanences_inc) - ((1 - correlation) * permanences_dec)
        # Get permanences for active neurons
        connection = self.permanences.data[neurons_active_mask]
        # Apply Hebbian update
        connection += update
        # Clamp to valid range
        connection.clamp_(0, 1)
        self.permanences.data[neurons_active_mask] = connection

    def forward(self, input_x, learn=True):
        """Forward pass with spatial pooling."""
        input_binary = self.encode_input(input_x)
        neurons_num_connected = self.compute_num_connected(input_binary)
        neurons_activation, neurons_activated_indices = self.neurons_compete(neurons_num_connected)
        if learn: self._learn(input_binary, neurons_activation)

        return neurons_activation, neurons_activated_indices

    def calc_similarity(self, activation, memory_activation):
        '''Jaccard similarity'''
        activation = set(activation)
        memory_activation = set(memory_activation)
        intersection = len(activation & memory_activation)
        union = len(activation | memory_activation)
        return intersection / union if union != 0 else 0.0

    def predict(self, new_activation, model_memories):
        new_activation = new_activation.flatten().tolist()
        similarities = []
        for memory in model_memories:
            memory_activation = memory['neurons_activation']

            similarity = self.calc_similarity(new_activation, memory_activation)
            similarities.append((similarity, memory['label']))

        top_overlap = sorted(similarities, reverse=True)[:5]
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
            model_memory.append({'neurons_activation': active_neurons_indices.flatten().tolist(), 'label': str(label.item())})
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
