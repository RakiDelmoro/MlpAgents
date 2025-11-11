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

class Neuron(nn.Module):
    def __init__(self, basal_size, apical_size, image_size=784, num_classes=10):
        super().__init__()

        self.basal_size = basal_size
        self.apical_size = apical_size
        self.image_size = image_size
        self.num_classes = num_classes

        self.basal_encoder = BasalEncoder(input_size=image_size, output_size=basal_size, sparsity=0.03)
        self.apical_encoder = ApicalEncoder(num_categories=num_classes, output_size=apical_size, sparsity=0.03)

        self.basal_synapses = nn.Parameter(torch.randn(basal_size, device='cuda') * 0.1)
        self.apical_synapses = nn.Parameter(torch.randn(apical_size, device='cuda') * 0.1)
        self.threshold = nn.Parameter(torch.tensor(0.3, device='cuda'))

    def update_params(self, basal_features, apical_features, neuron_actvity):
        with torch.no_grad():
            delta_basal = 0.01 * neuron_actvity.unsqueeze(1) * (basal_features - neuron_actvity.unsqueeze(1) * self.basal_synapses)
            delta_apical = 0.01 * neuron_actvity.unsqueeze(1) * (apical_features - neuron_actvity.unsqueeze(1) * self.apical_synapses)
        
            self.basal_synapses += delta_basal.mean(0)
            self.apical_synapses += delta_apical.mean(0)

    def training_phase(self, image, label):
        while True:
            basal_features = self.basal_encoder.encode(image)
            apical_features = self.apical_encoder.encode(label)
            
            basal_activation = torch.matmul(basal_features, self.basal_synapses)
            apical_activation = torch.matmul(apical_features, self.apical_synapses)
           
            coincidence = basal_activation * apical_activation
            neuron = torch.sigmoid(coincidence - self.threshold)

            self.update_params(basal_features, apical_features, neuron)
            # print(neuron)
            if neuron.item() > self.threshold: break

    def inference_phase(self, image):
        basal_features = self.basal_encoder.encode(image)
        basal_activation = torch.matmul(basal_features, self.basal_synapses)

        neuron_firing_rate = []
        for each in range(self.num_classes):
            apical_features = self.apical_encoder.encode(each).unsqueeze(0)
            apical_activation = torch.matmul(apical_features, self.apical_synapses)

            coincidence = basal_activation * apical_activation
            neuron = torch.sigmoid(coincidence - self.threshold)

            neuron_firing_rate.append(neuron.item())

        return torch.tensor(neuron_firing_rate).argmax().item()
    
    def runner(self, train_loader, test_loader):
        for train_image, train_label in train_loader:
            image = torch.tensor(train_image, device='cuda')
            label = torch.tensor(train_label, device='cuda')
            self.training_phase(image, label)

        correctness = []
        for test_image, test_label in test_loader:
            test_image = torch.tensor(test_image, device='cuda')
            test_label = torch.tensor(test_label, device='cuda')
            predicted = self.inference_phase(test_image)
            correctness.append(predicted == test_label.item())

        return sum(correctness) / len(correctness)
        
#         print(neuron_firing_rate)

# x_train = torch.randn(10000, 784, device='cuda')
# y_train = torch.randint(0, 10, size=(10000,), device='cuda')

# model = Neuron(basal_size=512, apical_size=512)

# for each in range(len(x_train)):
#     x = x_train[each].unsqueeze(0)
#     y = y_train[each].unsqueeze(0)
#     model.training_phase(x, y)

# print(y_train[0])
# print(model.inference_phase(x_train[0].unsqueeze(0)))
