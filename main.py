import torch
from model import CIFAR10CNN

# Test the model
model = CIFAR10CNN()
test_input = torch.randn(1, 3, 32, 32)  # Fake CIFAR image
output = model(test_input)
print(f"Output shape: {output.shape}")  # Should be [1, 10]
print(f"Output values: {output}")