import tqdm
import pickle
import torch
import numpy as np
import torch.nn.functional as functional
from Models.standard_mlp import StandardMLP
from Models.mlp_agents import MultiAgents, AgentMarginLoss

def mnist_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_samples = img_arr.shape[0]    
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        yield img_arr[indices[start_idx:end_idx]], label_arr[indices[start_idx:end_idx]]

def main():
    NUM_CLASSES = 10
    MAX_EPOCHS = 20
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    TOTAL_TRAINING_SAMPLES = 10000
    TOTAL_TESTING_SAMPLES = 100

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    model = MultiAgents(num_in_agents=2, num_out_agents=10, agents_feature_size=128, out_agents_dim_size=32)
    # model = StandardMLP()
    loss_fn = AgentMarginLoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=TOTAL_TESTING_SAMPLES, shuffle=True)

        # Training Loop
        train_loss = []
        train_batch_samples = TOTAL_TRAINING_SAMPLES // BATCH_SIZE
        for i, (batched_image, batched_label) in enumerate(train_loader):
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = functional.one_hot(torch.tensor(batched_label, device=DEVICE), num_classes=NUM_CLASSES).float()

            model_prediction = model(batched_image)
            loss = loss_fn(model_prediction, batched_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if i == train_batch_samples:
                break

        # Test Loop
        accuracies = []
        test_batch_samples = TOTAL_TESTING_SAMPLES // BATCH_SIZE
        for i, (batched_image, batched_label) in enumerate(test_loader):
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            model_pred_probabilities = model(batched_image)
            batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())
            if i == test_batch_samples:
                break

        train_loss = sum(train_loss) / len(train_loss)
        accuracies = sum(accuracies) / len(accuracies)

        t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')

    torch.save(model, f'multi_agents.pth')

main()
