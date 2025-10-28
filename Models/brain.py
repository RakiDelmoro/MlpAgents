import torch
import numpy as np
import torch.nn as nn
from collections import Counter

import torch
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, input_size: int, num_neurons: int, local_neuron_act_sparsity: float = 0.05,
                 synapse_threshold: float = 0.5, boost_strength: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.sparsity = local_neuron_act_sparsity
        self.synapse_threshold = synapse_threshold
        self.boost_strength = boost_strength

        self.receptive_field_size = 3
        self.input_h, self.input_w = 28, 28
        self.col_h, self.col_w = 32, 32

        # Build and store receptive field mask
        self.register_buffer('receptive_field_mask', self._build_receptive_field_connection())

        # Permanent synaptic connections (0-1), initialized with receptive field
        permanences_init = torch.rand(num_neurons, input_size, device='cuda') * 0.4 + 0.3
        permanences_init = permanences_init * self.receptive_field_mask
        self.permanences = nn.Parameter(permanences_init, requires_grad=False)

        # Boost factors for inactive neurons
        self.register_buffer('boost', torch.ones(num_neurons, device='cuda'))
        # Duty cycles
        self.register_buffer('active_duty_cycle', torch.zeros(num_neurons, device='cuda'))
        self.register_buffer('overlap_duty_cycle', torch.zeros(num_neurons, device='cuda'))

    def encode_input(self, x):
        """Convert input to binary using threshold."""
        return (x > 0.4).float()

    def local_inhibition(self, neurons_activation, inhibition_radius=3):
        """Apply local inhibition using unfold for efficiency."""
        batch_size = neurons_activation.shape[0]
        activation_as_2d = neurons_activation.view(batch_size, self.col_h, self.col_w)
        
        kernel_size = 2 * inhibition_radius + 1
        spatial_size = kernel_size * kernel_size
        target_neuron_num =  max(1, int(spatial_size * self.sparsity))

        # Unfold to get neighborhood patches efficiently
        pad = inhibition_radius
        padded = torch.nn.functional.pad(activation_as_2d, (pad, pad, pad, pad), mode='constant', value=-1e9)
        unfolded = torch.nn.functional.unfold(padded.unsqueeze(1), kernel_size=(kernel_size, kernel_size), padding=0)  # Shape: (batch_size, kernel_size^2, col_h*col_w)
        
        # Get top neuron activation values and their positions in each neighborhood
        topk_values, topk_indices = torch.topk(unfolded, k=min(target_neuron_num, kernel_size * kernel_size), dim=1, largest=True)        
        # Get the k-th largest value (threshold) for each position
        kth_value = topk_values[:, -1, :]  # (batch_size, col_h*col_w)
        kth_value_2d = kth_value.view(batch_size, self.col_h, self.col_w)

        # Neuron wins if its num connected >= k-th largest in its neighborhood and num connected > 0
        active_neurons_2d = (activation_as_2d >= kth_value_2d) & (activation_as_2d > 0)
        neurons_activation = active_neurons_2d.view(batch_size, -1).float()
        
        return neurons_activation

    def compute_num_connected(self, input_activation):
        """Compute connected scores between input and neuron synapses."""
        # Create binary connectivity from permanences
        connected = (self.permanences > self.synapse_threshold).float()
        # Overlap = number of connected synapses that are active
        overlap = torch.matmul(input_activation, connected.t())
        return overlap

    def _build_receptive_field_connection(self):
        """
        Create a mask defining which input bits can connect to each column
        based on receptive field topology.
        """
        column_size = self.col_h * self.col_w
        input_size = self.input_h * self.input_w
        
        # Calculate receptive field center offset
        rf_offset = self.receptive_field_size // 2
        
        # Scale factors from input to column space
        scale_h = self.input_h / self.col_h
        scale_w = self.input_w / self.col_w
        
        connection = torch.zeros(column_size, input_size, device='cuda')
        
        for col_idx in range(column_size):
            col_row = col_idx // self.col_w
            col_col = col_idx % self.col_w
            
            # Calculate receptive field center in input space
            center_h = int((col_row + 0.5) * scale_h)
            center_w = int((col_col + 0.5) * scale_w)
            
            # Define receptive field bounds
            h_min = max(0, center_h - rf_offset)
            h_max = min(self.input_h, center_h + rf_offset + 1)
            w_min = max(0, center_w - rf_offset)
            w_max = min(self.input_w, center_w + rf_offset + 1)
            
            # Set mask for this column's receptive field
            for h in range(h_min, h_max):
                for w in range(w_min, w_max):
                    input_idx = h * self.input_w + w
                    connection[col_idx, input_idx] = 1.0
        
        return connection
    
    def _update_duty_cycles(self, active, neurons_activation):
        """Update duty cycles based on raw overlap (before boosting)."""
        decay = 0.999
        
        # Average across batch
        active_mean = active.mean(dim=0)
        overlap_mean = (neurons_activation > 0).float().mean(dim=0)
        
        # Exponential moving average
        self.active_duty_cycle.mul_(decay).add_(active_mean, alpha=1 - decay)
        self.overlap_duty_cycle.mul_(decay).add_(overlap_mean, alpha=1 - decay)

    def _update_boost(self):
        """Update boost factors based on duty cycles."""
        target_duty_cycle = self.sparsity
        min_duty_cycle = 0.001
        
        # Clamp duty cycles to avoid division by zero
        overlap_dc = torch.clamp(self.overlap_duty_cycle, min=min_duty_cycle)
        active_dc = torch.clamp(self.active_duty_cycle, min=min_duty_cycle)
        
        # Boost = target / actual, scaled by boost_strength
        boost = target_duty_cycle / active_dc
        self.boost = torch.clamp(boost * self.boost_strength, min=1.0, max=10.0)

    def _learn(self, x: torch.Tensor, active_neurons: torch.Tensor):
        """Update permanences using Hebbian learning."""
        permanences_inc = 0.02
        permanences_dec = 0.002

        neurons_active_indices = torch.where(active_neurons > 0)[1]
        # Strengthen synapses that fired
        self.permanences.data[neurons_active_indices] += x.mean(dim=0) * permanences_inc
        # Weaken synapses that didn't fire
        inactive_inputs = 1 - x
        self.permanences.data[neurons_active_indices] -= inactive_inputs.mean(dim=0) * permanences_dec
        
        # Clamp permanences to valid range and apply receptive field mask
        self.permanences.data = torch.clamp(self.permanences.data, 0, 1)
        self.permanences.data = self.permanences.data * self.receptive_field_mask

    def forward(self, x: torch.Tensor, learn: bool = True):
        """Forward pass with spatial pooling."""
        input_binary = self.encode_input(x)
        
        # Compute raw overlap (before boosting)
        neurons_num_connected = self.compute_num_connected(input_binary)
        
        # Apply boost to overlap
        boosted_neurons_num_connected = neurons_num_connected * self.boost.unsqueeze(0)
        
        # Get active neurons via local inhibition
        neurons_activation = self.local_inhibition(boosted_neurons_num_connected, inhibition_radius=3)
        active_neurons_indices = torch.where(neurons_activation > 0)[1]
        
        if learn:
            self._update_duty_cycles(neurons_activation, neurons_num_connected)
            self._update_boost()
            self._learn(input_binary, neurons_activation)
        
        return neurons_activation, active_neurons_indices

    def predict(self, new_activation, model_memories):
        new_activation = new_activation.flatten().tolist()
        scores = []
        for memory in model_memories:
            score = len(set(new_activation) & set(memory['neurons_activation'])) / len(new_activation)
            scores.append((score, memory['label']))

        top_scores = sorted(scores, reverse=True)[:5]
        top_scores_labels = [label for _, label in top_scores]

        votes = Counter(top_scores_labels)
        return votes.most_common(1)[0][0]

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
        for image, label in test_loader:
            batched_image = torch.tensor(image, device='cuda')

            _, active_neurons_indices = self.forward(batched_image, learn=False)
            predicted = self.predict(active_neurons_indices, model_memory)

            correct = int(int(predicted) == label.item())
            correctness.append(correct)

        return sum(correctness) / len(correctness)
