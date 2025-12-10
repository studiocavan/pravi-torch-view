"""Matplotlib-based visualization utilities for model comparison."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def plot_model_comparison(model_results: Dict[str, Dict[str, Any]], save_path: str = None):
    """
    Create a matplotlib visualization comparing multiple models.

    Args:
        model_results: Dictionary mapping model names to their metrics
                      e.g., {'Model A': {'accuracy': 0.95, 'loss': 0.1, 'params': 1000}}
        save_path: Optional path to save the figure. If None, displays the plot.
    """
    if not model_results:
        print("No models to compare")
        return

    model_names = list(model_results.keys())
    n_models = len(model_names)

    # Extract metrics
    accuracies = [model_results[name].get('accuracy', 0) for name in model_names]
    losses = [model_results[name].get('loss', 0) for name in model_names]
    params = [model_results[name].get('params', 0) for name in model_names]

    # Find best model
    best_idx = np.argmax(accuracies)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Color scheme: highlight best model
    colors = ['#3498db' if i != best_idx else '#2ecc71' for i in range(n_models)]

    # 1. Accuracy comparison (bar chart)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontweight='bold')

    # Add star to best model
    ax1.text(best_idx, accuracies[best_idx] + 0.05, '⭐ BEST',
            ha='center', fontsize=12, fontweight='bold', color='#2ecc71')

    # 2. Loss comparison (bar chart)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, losses, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Loss (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontweight='bold')

    # 3. Parameter count comparison (bar chart)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_names, params, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    ax3.set_title('Model Size (# Parameters)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, param in zip(bars3, params):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 4. Accuracy vs Parameters scatter plot
    ax4 = axes[1, 1]
    scatter = ax4.scatter(params, accuracies, s=300, c=colors, alpha=0.8,
                         edgecolors='black', linewidths=2)

    # Add labels for each point
    for i, name in enumerate(model_names):
        ax4.annotate(name, (params[i], accuracies[i]),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontweight='bold')

    ax4.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy vs Model Complexity', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim([0, 1.0])

    # Add summary text box
    best_name = model_names[best_idx]
    summary_text = f"🏆 Best Model: {best_name}\nAccuracy: {accuracies[best_idx]:.2%}"
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3),
             fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig
