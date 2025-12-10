"""Model comparison and visualization utilities."""

import torch
from typing import Dict, Any


def compare_models(model_results: Dict[str, Dict[str, Any]]):
    """
    Compare multiple models and visualize their performance.

    Args:
        model_results: Dictionary mapping model names to their metrics
                      e.g., {'Model A': {'accuracy': 0.95, 'loss': 0.1, 'params': 1000}}
    """
    if not model_results:
        print("No models to compare")
        return

    print("\n" + "="*70)
    print("MODEL COMPARISON".center(70))
    print("="*70 + "\n")

    # Find best model by accuracy
    best_model = max(model_results.items(), key=lambda x: x[1].get('accuracy', 0))
    best_name = best_model[0]

    # Display each model
    for model_name, metrics in model_results.items():
        is_best = model_name == best_name

        # Header with indicator
        indicator = "⭐ BEST" if is_best else "  "
        print(f"{indicator} {model_name}")
        print("-" * 70)

        # Metrics
        accuracy = metrics.get('accuracy', 0)
        loss = metrics.get('loss', 0)
        params = metrics.get('params', 0)

        # Visual accuracy bar
        bar_length = 50
        filled = int(accuracy * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"  Accuracy: {accuracy:.2%}  [{bar}]")
        print(f"  Loss:     {loss:.4f}")
        print(f"  Params:   {params:,}")

        # Show additional metrics
        for key, value in metrics.items():
            if key not in ['accuracy', 'loss', 'params']:
                print(f"  {key.capitalize()}: {value}")

        print()

    # Summary
    print("="*70)
    print(f"🏆 Best Model: {best_name} (Accuracy: {best_model[1]['accuracy']:.2%})")
    print("="*70 + "\n")
