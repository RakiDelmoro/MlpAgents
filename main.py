import tqdm
import pickle
import torch
import numpy as np

def image_to_patches(image_arr, patch_width=None, patch_height=None):
    image_reshaped = image_arr.reshape(image_arr.shape[0], 28, 28)
    img_height = image_reshaped.shape[1]
    img_width = image_reshaped.shape[2]
    
    patch_height = img_height if patch_height is None else patch_height
    patch_width = img_width if patch_width is None else patch_width
    num_patches = (img_height // patch_height) * (img_width // patch_width)
    img_patches = np.stack(np.split(image_reshaped, num_patches, axis=2), axis=1)

    return img_patches

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

def model_runner(model, use_agents=False):
    MAX_EPOCHS = 20
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'

    with open('./dataset/mnist.pkl', 'rb') as f: ((train_images, train_labels), (test_images, test_labels), _) = pickle.load(f, encoding='latin1')
    assert train_images.shape[0] == train_labels.shape[0]
    assert test_images.shape[0] == test_labels.shape[0]
    assert train_images.shape[1] == test_images.shape[1] == IMAGE_HEIGHT*IMAGE_WIDTH

    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for _ in (t := tqdm.trange(MAX_EPOCHS)):
        train_loader = mnist_dataloader(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=True)

        # Training Loop
        train_loss = []
        # train_batch_samples = TOTAL_TRAINING_SAMPLES // BATCH_SIZE
        for i, (batched_image, batched_label) in enumerate(train_loader):
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            _, loss = model(batched_image, batched_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # Test Loop
        accuracies = []
        agents_accuracy = 0
        # test_batch_samples = TOTAL_TESTING_SAMPLES // BATCH_SIZE
        for i, (batched_image, batched_label) in enumerate(test_loader):
            batched_image = torch.tensor(batched_image, requires_grad=True, device=DEVICE)
            batched_label = torch.tensor(batched_label, device=DEVICE)

            if use_agents:
                each_agents_accuracy = model.infer_each_agent(batched_image, batched_label) 
                batch_accuracy = each_agents_accuracy[-1]
                agents_accuracy += each_agents_accuracy
            else:
                model_pred_probabilities, _ = model(batched_image)
                batch_accuracy = (model_pred_probabilities.argmax(axis=-1) == batched_label).float().mean()

            accuracies.append(batch_accuracy.item())

        if not use_agents:
            train_loss = sum(train_loss) / len(train_loss)
            accuracies = sum(accuracies) / len(accuracies)
            t.set_description(f'Loss: {train_loss:.4f} Accuracy: {accuracies:.4f}')
        else:
            train_loss = sum(train_loss) / len(train_loss)
            agents_accuracies = agents_accuracy / len(accuracies)
            rounded = list(map(lambda x: round(x, 4), agents_accuracies.tolist()))
            t.set_description(f'Loss: {train_loss:.4f} Agent accuracy: {rounded}')

    # torch.save(model, f'multi_agents.pth')
