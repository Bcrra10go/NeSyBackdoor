import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from celebA_semisupervised import CelebANet

def load_model(model_path):
    """Load the trained model."""
    model = CelebANet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_random_sample():
    """Get a random sample from CelebA dataset."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = CelebA(root='./data', target_type='attr', download=False, transform=transform)
    idx = np.random.randint(len(dataset))
    image, attributes = dataset[idx]
    
    # Convert attributes from -1/1 to 0/1
    attributes = (attributes + 1) // 2
    
    return image, attributes

def display_predictions(model, image, true_attributes):
    """Display the image and model predictions."""
    # Get model predictions
    with torch.no_grad():
        pred_probs = model(image.unsqueeze(0)).squeeze()
        predictions = (pred_probs > 0.5).float()
    
    # Attribute names (in order)
    attr_names = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    
    # Display image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title('Input Image')
    plt.axis('off')
    
    # Display predictions vs true attributes
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(attr_names))
    
    # Only show attributes where prediction differs from true value or confidence is high/low
    interesting_attrs = []
    for i, (pred, true, name) in enumerate(zip(pred_probs, true_attributes, attr_names)):
        if abs(pred - true) > 0.5 or pred > 0.8 or pred < 0.2:
            interesting_attrs.append((name, pred.item(), true.item()))
    
    # Sort by prediction confidence
    interesting_attrs.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
    
    # Display top 15 most interesting predictions
    plt.figure(figsize=(10, 8))
    names = [x[0] for x in interesting_attrs[:15]]
    preds = [x[1] for x in interesting_attrs[:15]]
    trues = [x[2] for x in interesting_attrs[:15]]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.barh(x - width/2, preds, width, label='Predicted', color='skyblue')
    plt.barh(x + width/2, trues, width, label='True', color='lightcoral')
    
    plt.yticks(x, [n.replace('_', ' ') for n in names])
    plt.xlabel('Probability / Value')
    plt.title('Model Predictions vs True Attributes')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage in Jupyter notebook:
"""
# Load model and test
model = load_model('path_to_your_model.pth')
image, attributes = get_random_sample()
display_predictions(model, image, attributes)
""" 