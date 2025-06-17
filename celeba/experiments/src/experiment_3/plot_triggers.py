import torch
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt
import numpy as np

# Configuration
CONFIG = {
    'image_size': 64,
    'trigger_sizes': [1, 5, 10],
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
}

def create_trigger(size):
    """Create a red trigger pattern of given size."""
    trigger = torch.zeros(3, size, size)
    trigger[0, :, :] = 1.0  # Red channel = 1
    trigger[1:, :, :] = 0.0  # Green and Blue channels = 0
    return trigger

def apply_trigger(image, trigger_size, position='center'):
    """Apply trigger pattern to image at specified position."""
    image = image.clone()
    trigger = create_trigger(trigger_size)
    
    if position == 'center':
        center_x = (CONFIG['image_size'] - trigger_size) // 2
        center_y = (CONFIG['image_size'] - trigger_size) // 2
        image[:, center_y:center_y + trigger_size, center_x:center_x + trigger_size] = trigger
    elif position == 'bottom_right':
        image[:, -trigger_size:, -trigger_size:] = trigger
    elif position == 'corners':
        # Top-left corner
        image[:, :trigger_size, :trigger_size] = trigger
        # Top-right corner
        image[:, :trigger_size, -trigger_size:] = trigger
        # Bottom-left corner
        image[:, -trigger_size:, :trigger_size] = trigger
        # Bottom-right corner
        image[:, -trigger_size:, -trigger_size:] = trigger
        # Center
        center_x = (CONFIG['image_size'] - trigger_size) // 2
        center_y = (CONFIG['image_size'] - trigger_size) // 2
        image[:, center_y:center_y + trigger_size, center_x:center_x + trigger_size] = trigger
    
    return image

def main():
    # Load CelebA dataset
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.CenterCrop(CONFIG['image_size']),
        transforms.ToTensor(),
    ])
    
    dataset = CelebA(root='../../../data', split='train', transform=transform, download=True)
    
    # Get a single image
    image, _ = dataset[0]
    
    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('CelebA Image with Different Trigger Patterns', fontsize=16)
    
    # Plot images with different triggers
    positions = [
        ('1x1 Center', 1, 'center'),
        ('1x1 Bottom Right', 1, 'bottom_right'),
        ('1x1 Corners', 1, 'corners'),
        ('5x5 Center', 5, 'center'),
        ('5x5 Bottom Right', 5, 'bottom_right'),
        ('5x5 Corners', 5, 'corners'),
        ('10x10 Center', 10, 'center'),
        ('10x10 Bottom Right', 10, 'bottom_right'),
        ('10x10 Corners', 10, 'corners'),
    ]
    
    for idx, (title, size, pos) in enumerate(positions):
        row = idx // 3
        col = idx % 3
        triggered_image = apply_trigger(image, size, pos)
        axes[row, col].imshow(triggered_image.permute(1, 2, 0))
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('trigger_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main() 