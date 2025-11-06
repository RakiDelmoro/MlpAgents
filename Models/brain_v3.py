import torch
import torch.nn as nn

class BasalEncoder:
    def __init__(self, input_size, output_size, sparsity=0.03, seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.num_active = int(round(output_size * sparsity))
        
        # Initialize random projection for encoding
        torch.manual_seed(seed)
        self.projection = torch.randn(input_size, output_size, device='cuda') * 0.1
    
    def encode(self, x):
        # Handle single sample vs batch
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        # Project to output space
        activations = torch.matmul(x, self.projection)
        
        # Get top-k indices for sparsification
        _, top_indices = torch.topk(activations, self.num_active, dim=1)
        
        # Create sparse representation
        sdr = torch.zeros_like(activations, device='cuda')
        sdr.scatter_(1, top_indices, 1.0)
        
        if squeeze:
            sdr = sdr.squeeze(0)
        
        return sdr

class ApicalEncoder:    
    def __init__(self, num_categories, output_size, sparsity=0.03, seed=42):
        self.num_categories = num_categories
        self.output_size = output_size
        self.sparsity = sparsity
        self.num_active = int(round(output_size * sparsity))
        
        # Pre-generate fixed SDR for each category
        torch.manual_seed(seed)
        self.category_sdrs = self._generate_category_sdrs()
    
    def _generate_category_sdrs(self):
        """Generate non-overlapping SDRs for each category"""
        sdrs = torch.zeros(self.num_categories, self.output_size, device='cuda')
        
        # Calculate bucket size per category to minimize overlap
        bucket_size = self.output_size // self.num_categories
        
        for i in range(self.num_categories):
            # Sample active bits from category's bucket with some overlap allowance
            start_idx = i * bucket_size
            end_idx = min(start_idx + bucket_size + self.num_active, self.output_size)
            
            # Randomly select active bits within the bucket
            available_indices = torch.arange(start_idx, end_idx)
            perm = torch.randperm(len(available_indices))
            active_indices = available_indices[perm[:self.num_active]]
            
            sdrs[i, active_indices] = 1.0
        
        return sdrs
    
    def encode(self, labels):
        if isinstance(labels, int):
            return self.category_sdrs[labels]
        
        squeeze = False
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
            squeeze = True
        
        sdr = self.category_sdrs[labels]
        
        if squeeze:
            sdr = sdr.squeeze(0)
        
        return sdr
    
class CorticalColumn(nn.Module):
    def __init__(self, basal_size=512, apical_size=512, input_image_size=784, 
                 num_classes=10, neurons_size=1024, sparsity=0.03, 
                 synaptic_threshold=0.1):
        super().__init__()
        self.sparsity = sparsity
        self.basal_size = basal_size
        self.apical_size = apical_size
        self.neurons_size = neurons_size
        self.synaptic_threshold = synaptic_threshold
        self.num_classes = num_classes

        # Initialize encoders
        self.basal_encoder = BasalEncoder(input_size=input_image_size, output_size=basal_size, sparsity=sparsity)
        self.apical_encoder = ApicalEncoder(num_categories=num_classes, output_size=apical_size, sparsity=sparsity)

        # Synaptic connections
        self.basal_synapse = nn.Parameter(torch.rand(basal_size, neurons_size, device='cuda') * 0.4 + 0.3, requires_grad=False)
        self.apical_synapse = nn.Parameter(torch.rand(apical_size, neurons_size, device='cuda') * 0.4 + 0.3,requires_grad=False)

        self.num_active_neurons = int(round(neurons_size * sparsity))

    def get_total_params(self):
        return self.basal_synapse.flatten().shape[0] + self.apical_synapse.flatten().shape[0]
    
    def get_neuron_firing(self, neurons_activation):
        top_indices = torch.topk(neurons_activation, self.num_active_neurons, dim=-1)[1]
        neurons_firing = torch.zeros_like(neurons_activation)
        neurons_firing.scatter_(-1, top_indices, 1.0)

        return neurons_firing
    
    def basal_update(self, basal_features, neurons_activation):
        pre_neuron_activation = basal_features.T
        post_neuron_activation = neurons_activation

        hebbian = torch.matmul(pre_neuron_activation, post_neuron_activation)
        decay = self.basal_synapse * (post_neuron_activation ** 2)
        self.basal_synapse.data += 0.01 * (hebbian - decay)

        self.basal_synapse.data.clamp_(0.0, 1.0)

    def apical_update(self, apical_features, neurons_activation, neighborhood_size=5):        
        with torch.no_grad():            
            # Compute neighborhood strength (cooperative signal)
            neighborhood_strength = torch.zeros_like(self.apical_synapse)
            
            for i in range(self.apical_size):
                start = max(0, i - neighborhood_size // 2)
                end = min(self.apical_size, i + neighborhood_size // 2 + 1)
                neighborhood_strength[i, :] = self.apical_synapse[start:end, :].mean(dim=0)

            hebbian_term = torch.matmul(apical_features.T, neurons_activation)            
            # Cooperative modulation: strengthen more if neighbors are strong
            cooperative_hebbian = hebbian_term * neighborhood_strength
            output_activity = (neurons_activation ** 2)
            oja_decay = self.apical_synapse * output_activity
    
            self.apical_synapse.data += 0.01 * (cooperative_hebbian - oja_decay)
            self.apical_synapse.data.clamp_(0.0, 1.0)

    def training_phase(self, image, label):
        basal_features = self.basal_encoder.encode(image)
        apical_features = self.apical_encoder.encode(label)

        basal_activation = torch.matmul(basal_features, self.basal_synapse)
        apical_activation = torch.matmul(apical_features, self.apical_synapse)
        neurons_activation = self.get_neuron_firing(basal_activation + apical_activation)

        self.basal_update(basal_features, neurons_activation)
        self.apical_update(apical_features, neurons_activation)

    def inference_phase(self, image):
        basal_features = self.basal_encoder.encode(image)
        basal_activation = self.get_neuron_firing(torch.matmul(basal_features, self.basal_synapse))
        similarities = []
        for digit in range(self.num_classes):
            apical_features = self.apical_encoder.encode(digit).unsqueeze(0)
            apical_activation = self.get_neuron_firing(torch.matmul(apical_features, self.apical_synapse))
            similarity = torch.nn.functional.cosine_similarity(basal_activation, apical_activation, dim=-1)
            similarities.append(similarity)
        
        similarities = torch.stack(similarities)
        predicted_label = torch.argmax(similarities).item()

        return predicted_label
    
    def runner(self, train_loader, test_loader):
        for train_image, train_label in train_loader:
            train_image = torch.tensor(train_image, device='cuda')
            train_label = torch.tensor(train_label, device='cuda')
            self.training_phase(train_image, train_label)

        correctness = []
        for test_image, test_label in test_loader:
            test_image = torch.tensor(test_image, device='cuda')
            test_label = torch.tensor(test_label, device='cuda')
            predicted = self.inference_phase(test_image)
            correctness.append(predicted == test_label.item())

        return sum(correctness) / len(correctness)
