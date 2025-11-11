# Neural Nets from Scratch

A simple, educational neural network library implemented from scratch using NumPy. This project demonstrates the fundamentals of deep learning by implementing core components like layers, activations, loss functions, and backpropagation.

## Features

### Models

- **Linear**: Fully connected layer
- **Sequential**: Container for chaining multiple layers together

### Activations

- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid activation function
- **Softmax**: Softmax activation for multi-class classification
- **Tanh**: Hyperbolic tangent activation

### Loss Functions

- **L1Loss**: Mean Absolute Error
- **L2Loss**: Mean Squared Error
- **BinaryCrossEntropyLoss**: Binary cross-entropy for binary classification
- **CategoricalCrossEntropyLoss**: Cross-entropy for multi-class classification

## Installation

### Setup Virtual Environment

```bash
# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### Install Dependencies

```bash
# Make sure venv is activated (you should see (.venv) in your prompt)
pip install -r requirements.txt
```

> **Visualization dependencies:** all animations rely on `matplotlib` and `pillow`. The loss landscape animation additionally requires `scipy>=1.13` (already included in `requirements.txt`). If SciPy is missing, only the loss landscape helper is unavailable.

## Usage

### Training a Model

Train a neural network on generated datasets (circles or moons):

```bash
# Basic training
python scripts/train_generated_datasets.py

# Train with video generation (creates MP4/GIF videos)
python scripts/train_generated_datasets.py --record

# Train on circles dataset with video generation
python scripts/train_generated_datasets.py --record --dataset-type circles

# Train on moons dataset with video generation
python scripts/train_generated_datasets.py --record --dataset-type moons
```

**Command-line Options:**

- `--record` or `--use-recording`: Enable recording for video generation
- `--dataset-type`: Choose dataset type (`circles` or `moons`, default: `moons`)

**Note:** For MP4 video generation, install FFmpeg:

```bash
brew install ffmpeg  # macOS
# or
sudo apt-get install ffmpeg  # Linux
```

If FFmpeg is not available, videos will be saved as GIF files instead.

## Project Structure

```
src/nn_library/
├── models/          # Neural network layers
├── activation/      # Activation functions
├── loss/           # Loss functions
└── visualization/  # Visualization tools
```

## Key Design Principles

- **Modular**: Each component (layer, activation, loss) is independent
- **Educational**: Clear implementation to understand how neural networks work
- **From Scratch**: Built using only NumPy, no deep learning frameworks
- **Backpropagation**: Full implementation of automatic differentiation

## Visualization Tools

The library includes visualization tools:

- **Decision boundaires** - Animated videos of decision boundary evolution during training
- **Neuron activations** - Understand what each neuron represents
- **Loss landscape exploration** - Project optimization progress into a 2D slice of parameter space

### Available animations

| Function                             | What it shows                                         | Inputs you need                                     | Extra requirements       |
| ------------------------------------ | ----------------------------------------------------- | --------------------------------------------------- | ------------------------ |
| `animate_decision_boundary_training` | Evolution of the 2D decision boundary during training | Trained model, 2D data, `TrainingRecorder` history  | —                        |
| `animate_dataset_scatter_plot`       | Progressive reveal of the input dataset               | 2D dataset                                          | —                        |
| `animate_probability_distribution`   | Model confidence for each training sample over time   | Trained model, dataset, `TrainingRecorder` history  | —                        |
| `animate_training_loss`              | Train vs. test loss curves                            | `TrainingRecorder` with `loss` and `test_loss`      | —                        |
| `animate_training_accuracy`          | Train vs. test accuracy curves                        | `TrainingRecorder` with `accuracy` metrics          | —                        |
| `animate_network_architecture`       | Network graph with live activations                   | Trained model, sample batch, `TrainingRecorder`     | —                        |
| `animate_loss_landscape_2d`          | 2D slice of the loss surface with trajectory          | Trained model, dataset, loss fn, `TrainingRecorder` | Requires `scipy >= 1.13` |

### Video Generation

Training videos are automatically generated when using the `--record` flag. The videos show how the decision boundary evolves as the model learns, with each frame representing a training epoch.

Videos are saved to the `visualizations/` directory with names like:

- `decision_boundary_training_circles.mp4`
- `decision_boundary_training_moons.mp4`

You can also use the visualization API directly:

```python
from visualization import TrainingRecorder, animate_decision_boundary_training

# During training
recorder = TrainingRecorder(record_every_n_epochs=1)
# ... train model, call recorder.record() each epoch ...

# Generate video
animate_decision_boundary_training(model, X_train, y_train, recorder,
                                  save_path='training_video.mp4')
```

## License

This is an educational project for learning purposes. Free to use. Please credit me!
