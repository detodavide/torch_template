# PyTorch Training and Inference Runner (TIR)

The PyTorch Training and Inference Runner (TIR) is a Python script designed to streamline the process of training, validating, checkpointing, and making predictions using PyTorch models. It provides a class called `TIR` that encapsulates common functionalities required during the machine learning lifecycle.

## Features

- Efficient training and validation loops.
- Checkpointing and resuming training from a saved checkpoint.
- Predictions using trained models.
- Loss visualization through a plot.
- Reproducibility settings for consistent results.

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

Install the required packages using the following command:

```python
pip install torch numpy matplotlib
```

## How to Use

1. Import the necessary modules and classes at the beginning of your script:

```python
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
```

2. Copy and paste the `TIR` class definition into your script.

3. Create your PyTorch model, loss function, and optimizer.

4. Prepare your training and validation data loaders.

5. Create an instance of the `TIR` class, passing your model, loss function, and optimizer as arguments:

```python
tir = TIR(model, loss_fn, optimizer)
```

6. Set the training and validation data loaders using the `set_loaders` method:

```python
tir.set_loaders(train_loader, val_loader)
```

7. Train your model using the `train` method:

```python
tir.train(n_epochs=10)
```

8. Optionally, save and load checkpoints using the `save_checkpoint` and `load_checkpoint` methods.

9. Make predictions using the `predict` method:

```python
predictions = tir.predict(input_data)
```

10. Visualize training and validation losses using the `plot_losses` method:

```python
tir.plot_losses()
plt.show()
```

## Example Usage

```python
# Example usage of TIR for training a PyTorch model

# Import necessary modules

# Define your PyTorch model, loss function, and optimizer

# Prepare your training and validation data loaders

# Create an instance of TIR
tir = TIR(model, loss_fn, optimizer)

# Set the data loaders
tir.set_loaders(train_loader, val_loader)

# Train the model
tir.train(n_epochs=10)

# Make predictions
predictions = tir.predict(input_data)

# Plot losses
tir.plot_losses()
plt.show()
```