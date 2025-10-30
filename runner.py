import tqdm
import pickle
import torch
import numpy as np
from Models.brain import Agent

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

def training_runner():
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    model = Agent(input_size=(28,28), receptive_field_size=(28, 4), num_neurons=1024)
    for epoch in range(1, 101):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=1, shuffle=True)
        # LEARNING...
        model.phase_1(train_loader)
        # Test model every 1 epochs
        if epoch % 1  == 0:
            memory_loader = mnist_dataloader(train_images, train_labels, batch_size=1, shuffle=True)
            test_loader = mnist_dataloader(test_images, test_labels, batch_size=1, shuffle=True)

            model_memories = model.phase_2(memory_loader)
            accuracy, scores = model.test(model_memories, test_loader)
            print(f'EPOCH: {epoch} Accuracy: {accuracy} Average score: {scores}')
        else:
            print(f'EPOCH: {epoch} not testing yet...')

    torch.save(model, 'RockV1.pth')

training_runner()
