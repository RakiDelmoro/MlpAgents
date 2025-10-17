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
    prediction, _ = model(torch.tensor(normalized_arr, dtype=torch.float32, device='cuda'))
    print(f'MODEL: {split[0]} Predicted: {prediction.argmax(-1).item()}')

def test_multi_agent_model(model, model_path: str):
    # Load Image
    split = model_path.split('.')

    img = Image.open('test_image.png')
    image_as_arr = np.asarray(img).reshape(1, 28*28)
    # normalize pixel value to range [0, 1]
    normalized_arr = (image_as_arr - np.min(image_as_arr)) / (np.max(image_as_arr) - np.min(image_as_arr))
    
    model.eval()
    each_agents_prediction = model.test_each_agent(torch.tensor(normalized_arr, dtype=torch.float32, device='cuda'))
    print(f'MODEL: {split[0]} Predicted: {each_agents_prediction}')

multi_agents = torch.load('multi_agents_v2.pth')
standard_mlp = torch.load('standard_mlp.pth')

test_multi_agent_model(multi_agents, 'multi_agents_v2.pth')
test_model(standard_mlp, 'standard_mlp.pth')
