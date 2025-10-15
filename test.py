import torch
import numpy as np
from PIL import Image

def test_model(model, model_path: str):
    # Load Image
    split = model_path.split('.')

    img = Image.open('test_image.png')
    image_as_arr = np.asarray(img).reshape(1, 28*28)
    # normalize pixel value to range [0, 1]
    normalized_arr = (image_as_arr - np.min(image_as_arr)) / (np.max(image_as_arr) - np.min(image_as_arr))

    # Model prediction
    prediction = model(torch.tensor(normalized_arr, dtype=torch.float32, device='cuda'))
    print(f'MODEL: {split[0]} Predicted: {prediction.argmax(-1).item()}')

multi_agents = torch.load('multi_agents.pth')
standard_mlp = torch.load('mlp.pth')

test_model(multi_agents, 'multi_agents.pth')
test_model(standard_mlp, 'mlp.pth')
