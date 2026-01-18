"""
Utility functions for transformer learning notebooks.
Includes visualization helpers, common operations, and debugging tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import torch


def set_style():
    """Set consistent plotting style for all notebooks."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_attention_weights(attention,
                          tokens: Optional[List[str]] = None,
                          title: str = "Attention Weights",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention: Attention weight matrix (seq_len, seq_len) or (batch, heads, seq_len, seq_len)
        tokens: Optional list of token strings for axis labels
        title: Plot title
        figsize: Figure size tuple
    """
    # Handle multi-dimensional attention (take first batch, first head)
    if len(attention.shape) == 4:
        attention = attention[0, 0]
    elif len(attention.shape) == 3:
        attention = attention[0]

    # Convert to numpy if torch tensor
    if torch.is_tensor(attention):
        attention = attention.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(attention,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                xticklabels=tokens if tokens else 'auto',
                yticklabels=tokens if tokens else 'auto',
                cbar_kws={'label': 'Attention Weight'})
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()


def plot_multi_head_attention(attention,
                              num_heads: int = 8,
                              tokens: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (16, 12)):
    """
    Visualize multiple attention heads in a grid.

    Args:
        attention: Attention weights (batch, heads, seq_len, seq_len)
        num_heads: Number of attention heads to display
        tokens: Optional list of token strings
        figsize: Figure size tuple
    """
    if torch.is_tensor(attention):
        attention = attention.detach().cpu().numpy()

    # Take first batch if needed
    if len(attention.shape) == 4:
        attention = attention[0]

    rows = int(np.ceil(num_heads / 4))
    fig, axes = plt.subplots(rows, 4, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_heads):
        sns.heatmap(attention[i],
                   ax=axes[i],
                   cmap='viridis',
                   xticklabels=tokens if tokens else False,
                   yticklabels=tokens if tokens else False,
                   cbar=True)
        axes[i].set_title(f'Head {i+1}')

    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_embeddings_2d(embeddings,
                       labels: Optional[List[str]] = None,
                       method: str = 'pca',
                       title: str = "Embedding Visualization"):
    """
    Visualize high-dimensional embeddings in 2D using PCA or t-SNE.

    Args:
        embeddings: Embedding matrix (n_samples, embedding_dim)
        labels: Optional labels for each point
        method: 'pca' or 'tsne'
        title: Plot title
    """
    from sklearn.decomposition import PCA

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        title += f"\nPCA (variance explained: {sum(explained_var):.1%})"
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label,
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=9,
                        alpha=0.7)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_curves(losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         title: str = "Training Progress"):
    """
    Plot training and validation loss curves.

    Args:
        losses: List of training losses
        val_losses: Optional list of validation losses
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', linewidth=2)

    if val_losses:
        plt.plot(val_losses, label='Validation Loss', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_outputs(numpy_output, pytorch_output, tolerance=1e-5):
    """
    Compare NumPy and PyTorch implementations to verify equivalence.

    Args:
        numpy_output: Output from NumPy implementation
        pytorch_output: Output from PyTorch implementation
        tolerance: Maximum allowed difference

    Returns:
        bool: True if outputs match within tolerance
    """
    if torch.is_tensor(pytorch_output):
        pytorch_output = pytorch_output.detach().cpu().numpy()

    diff = np.abs(numpy_output - pytorch_output)
    max_diff = np.max(diff)

    print(f"Maximum difference: {max_diff:.2e}")

    if max_diff < tolerance:
        print(f"✓ Outputs match within tolerance ({tolerance:.2e})")
        return True
    else:
        print(f"✗ Outputs differ by more than tolerance ({tolerance:.2e})")
        print(f"  Mean difference: {np.mean(diff):.2e}")
        print(f"  Std difference: {np.std(diff):.2e}")
        return False


def print_tensor_info(tensor, name: str = "Tensor"):
    """
    Print detailed information about a tensor for debugging.

    Args:
        tensor: NumPy array or PyTorch tensor
        name: Name to display
    """
    if torch.is_tensor(tensor):
        print(f"\n{name} (PyTorch Tensor):")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Requires grad: {tensor.requires_grad}")
        tensor_np = tensor.detach().cpu().numpy()
    else:
        print(f"\n{name} (NumPy Array):")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        tensor_np = tensor

    print(f"  Min: {tensor_np.min():.4f}")
    print(f"  Max: {tensor_np.max():.4f}")
    print(f"  Mean: {tensor_np.mean():.4f}")
    print(f"  Std: {tensor_np.std():.4f}")


def softmax(x, axis=-1):
    """
    Numerically stable softmax implementation.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_padding_mask(seq_len: int, pad_indices: List[int]) -> np.ndarray:
    """
    Create a padding mask for attention.

    Args:
        seq_len: Sequence length
        pad_indices: Indices to mask (padding positions)

    Returns:
        Mask array where 1 = valid, 0 = padding
    """
    mask = np.ones((seq_len,))
    mask[pad_indices] = 0
    return mask


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (look-ahead) mask for autoregressive attention.

    Args:
        seq_len: Sequence length

    Returns:
        Lower triangular mask (seq_len, seq_len)
    """
    return np.tril(np.ones((seq_len, seq_len)))


def visualize_causal_mask(seq_len: int = 8):
    """
    Visualize a causal attention mask.

    Args:
        seq_len: Sequence length to visualize
    """
    mask = create_causal_mask(seq_len)

    plt.figure(figsize=(8, 8))
    sns.heatmap(mask,
                annot=True,
                fmt='.0f',
                cmap='RdYlGn',
                cbar_kws={'label': '1=Allowed, 0=Masked'},
                square=True)
    plt.title('Causal Attention Mask\n(each position can only attend to previous positions)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick test of visualization functions
    print("Testing utility functions...")

    # Test attention visualization
    sample_attention = np.random.rand(6, 6)
    sample_attention = sample_attention / sample_attention.sum(axis=1, keepdims=True)

    set_style()
    plot_attention_weights(sample_attention,
                          tokens=['I', 'love', 'learning', 'about', 'transformers', '!'],
                          title="Sample Attention Pattern")

    # Test causal mask
    visualize_causal_mask(8)

    print("\n✓ Utils module loaded successfully!")
