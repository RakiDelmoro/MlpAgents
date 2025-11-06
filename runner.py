import tqdm
import pickle
import torch
import numpy as np
from Models.brain import Agent
import matplotlib.pyplot as plt
# from Models.brain_v2 import Agent
from Models.brain_v3 import CorticalColumn

def plot_training_progress(mlp_accuracies, agent_accuracies, filename='training_progress.png'):
    epochs = list(range(len(mlp_accuracies)))  # x-axis = epoch numbers

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mlp_accuracies, label='MLP Accuracy', color='blue', marker='o')
    plt.plot(epochs, agent_accuracies, label='Agent Accuracy', color='orange', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress: Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot to file
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✅ Training progress plot saved as '{filename}'")

def mnist_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_samples = img_arr.shape[0]    
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size

        # img_arr = image_to_patches(img_arr, patch_width=14)
        batched_img = img_arr[indices[start_idx:end_idx]]
        batched_label = label_arr[indices[start_idx:end_idx]]

        yield batched_img, batched_label

def standard_mlp(model, train_loader, test_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Training
    for image, label in train_loader:
        image = torch.tensor(image, device='cuda')
        label = torch.tensor(label, device='cuda')

        model_pred = model(image)
        loss = loss_fn(model_pred, label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correctness = []
    for test_image, test_label in test_loader:
        test_image = torch.tensor(test_image, device='cuda')
        test_label = torch.tensor(test_label, device='cuda')

        model_pred = model(test_image).argmax(dim=-1).item()
        correctness.append(model_pred == test_label.item())

    return sum(correctness) / len(correctness)

def training_runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH
    agent = CorticalColumn(neurons_size=768)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(784, 1024, device='cuda')
            self.layer2 = torch.nn.Linear(1024, 10, device='cuda')
        def forward(self, input_x):
            return self.layer2(self.layer1(input_x).relu())

    mlp = Mlp()

    mlp_accuracies = []
    agent_accuracies = []
    for epoch in range(1, 101):
        mlp_train_loader = mnist_dataloader(train_images[:100], train_labels[:100], batch_size=1, shuffle=True)
        mlp_test_loader = mnist_dataloader(test_images[:100], test_labels[:100], batch_size=1, shuffle=False)

        agent_train_loader = mnist_dataloader(train_images[:100], train_labels[:100], batch_size=1, shuffle=True)
        agent_test_loader =  mnist_dataloader(test_images[:100], test_labels[:100], batch_size=1, shuffle=False)

        agent_accuracy = agent.runner(agent_train_loader, agent_test_loader)
        mlp_accuracy = standard_mlp(mlp, mlp_train_loader, mlp_test_loader)

        mlp_accuracies.append(mlp_accuracy)
        agent_accuracies.append(agent_accuracy)

        print(f'EPOCH: {epoch}: MLP: {mlp_accuracy} AGENT: {agent_accuracy}')

        # After training completes:
        plot_training_progress(mlp_accuracies, agent_accuracies, 'training_progress.png')

training_runner()
