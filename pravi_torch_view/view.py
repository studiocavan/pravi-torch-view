"""Core visualization functions for PyTorch tensors."""

import torch


def view(tensor):
    """
    Display detailed information about a PyTorch tensor.

    Args:
        tensor: PyTorch tensor to visualize
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    print(f"Tensor Information:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires Grad: {tensor.requires_grad}")
    print(f"  Memory: {tensor.element_size() * tensor.nelement()} bytes")
    print(f"\n{tensor}")
