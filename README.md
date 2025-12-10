# pravi-torch-view

A PyTorch tensor visualization tool.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Usage

### Basic Tensor Visualization

```python
import torch
from pravi_torch_view import view

# Create a tensor
tensor = torch.randn(3, 4, 5)

# View tensor information
view(tensor)
```

### Model Comparison

Compare multiple models to see which performs better:

```python
from pravi_torch_view import compare_models

# Define your model results
results = {
    'Shallow CNN': {
        'accuracy': 0.87,
        'loss': 0.421,
        'params': 31370,
        'layers': 1,
    },
    'Deep CNN': {
        'accuracy': 0.94,
        'loss': 0.198,
        'params': 103050,
        'layers': 2,
    }
}

# Visualize the comparison
compare_models(results)
```

**Output:**
```
======================================================================
                          MODEL COMPARISON
======================================================================

     Shallow CNN
----------------------------------------------------------------------
  Accuracy: 87.00%  [███████████████████████████████████████████░░░░░░░]
  Loss:     0.4210
  Params:   31,370
  Layers:   1

⭐ BEST Deep CNN
----------------------------------------------------------------------
  Accuracy: 94.00%  [███████████████████████████████████████████████░░░]
  Loss:     0.1980
  Params:   103,050
  Layers:   2

======================================================================
🏆 Best Model: Deep CNN (Accuracy: 94.00%)
======================================================================
```

### Matplotlib Visualization

Create beautiful graphical visualizations using matplotlib:

```python
from pravi_torch_view import plot_model_comparison

# Same results dictionary as above
results = {
    'Shallow CNN': {'accuracy': 0.87, 'loss': 0.421, 'params': 31370},
    'Deep CNN': {'accuracy': 0.94, 'loss': 0.198, 'params': 103050}
}

# Display interactive plot
plot_model_comparison(results)

# Or save to file
plot_model_comparison(results, save_path='comparison.png')
```

This creates a comprehensive 4-panel visualization showing:
- **Accuracy comparison** with the best model highlighted
- **Loss comparison** (lower is better)
- **Model size** (parameter count)
- **Accuracy vs Complexity** scatter plot

### Real-World Example

See a complete working example comparing two CNN models:

```bash
python examples/model_comparison_example.py
```

This example trains and compares a shallow vs. deep CNN on synthetic MNIST-like data, showing both text and graphical visualizations.

## Development

This project is in early development.
