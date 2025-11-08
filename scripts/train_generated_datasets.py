import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from nn_library.activation import Sigmoid, Tanh, ReLU
from nn_library.loss import BinaryCrossEntropyLoss
from nn_library.models import Sequential, Linear
from visualization import TrainingRecorder, animate_decision_boundary_training

from sklearn.datasets import make_circles, make_moons
import numpy as np

# X_batch and y_batch are already batched
# We can expect these inputs to be (batch_size, feature_dim)
# or (batch_size, output_dim)
def train_step(X_batch, y_batch, model, loss_fn, lr):
    """Single training step."""
    y_pred = model.forward(X_batch)
    loss = loss_fn.calculate(y_pred, y_batch)
    gradient = loss_fn.get_gradient()
    model.backward(gradient)
    model.step(lr)
    return loss


def load_circles_dataset(n_samples=1000, noise=0.25):
    X, y = make_circles(n_samples=n_samples, noise=noise)
    return X, y


def load_moons_dataset(n_samples=1000, noise=0.25):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y


def accuracy(X, y, model):
    y_pred = model.forward(X)
    y_pred = np.round(y_pred)
    return np.mean(y_pred == y)


def test(X_test, y_test, model, loss_fn):
    y_pred = model.forward(X_test)
    loss = loss_fn.calculate(y_pred, y_test)
    return loss


def train(model, loss_fn, lr, X_train, y_train, X_test, y_test, epochs, batch_size, recorder=None):
    """Train model with optional recording for visualization."""
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            loss = train_step(X_batch, y_batch, model, loss_fn, lr)
            epoch_losses.append(loss)
        
        # Average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        # Compute accuracies
        train_acc = accuracy(X_train, y_train, model)
        test_acc = accuracy(X_test, y_test, model)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Test loss
        test_loss = test(X_test, y_test, model, loss_fn)
        test_losses.append(test_loss)
        
        # Record state if recorder is provided
        if recorder is not None:
            recorder.record(model, epoch, avg_loss, accuracy=train_acc, 
                          test_loss=test_loss, test_accuracy=test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}, Train Acc: {train_acc:.2%}, "
                  f"Test Acc: {test_acc:.2%}")
    
    return train_losses, test_losses, train_accuracies, test_accuracies


def main():
    parser = argparse.ArgumentParser(description='Train a neural network from scratch')
    parser.add_argument('--record', '--use-recording', action='store_true',
                        help='Enable recording for video generation')
    parser.add_argument('--dataset-type', type=str, default='moons',
                        help='Type of dataset to use (circles or moons)')
    args = parser.parse_args()
    
    # Dataset configuration
    dataset_type = args.dataset_type
    n_samples_train = 1024
    n_samples_test = 256
    noise = 0.1
    
    # Training configuration
    use_recording = args.record
    lr = 0.05
    epochs = 150
    batch_size = 32
    
    # Model architecture
    input_dim = 2  # Change to 3 if adding radius feature
    model_layers = [
        Linear(input_dim, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    ]
    
    print("Generating dataset...")
    if dataset_type == 'circles':
        X_train, y_train = load_circles_dataset(n_samples=n_samples_train, noise=noise)
        X_test, y_test = load_circles_dataset(n_samples=n_samples_test, noise=noise)
    elif dataset_type == 'moons':
        X_train, y_train = load_moons_dataset(n_samples=n_samples_train, noise=noise)
        X_test, y_test = load_moons_dataset(n_samples=n_samples_test, noise=noise)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    # Optional: Add third dimension (radius)
    # Uncomment to add radius feature
    # X_train = np.hstack((X_train, np.sqrt(X_train[:, 0]**2 + X_train[:, 1]**2).reshape(-1, 1)))
    # X_test = np.hstack((X_test, np.sqrt(X_test[:, 0]**2 + X_test[:, 1]**2).reshape(-1, 1)))
    # input_dim = 3
    
    # Reshape labels
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Create model
    model = Sequential(model_layers)
    loss_fn = BinaryCrossEntropyLoss()
    
    # Setup recorder if visualization is enabled
    recorder = None
    if use_recording:
        recorder = TrainingRecorder(record_every_n_epochs=1)
        print("Recording enabled for video generation")
    
    # Initial metrics
    initial_train_loss = test(X_train, y_train, model, loss_fn)
    initial_test_loss = test(X_test, y_test, model, loss_fn)
    initial_train_accuracy = accuracy(X_train, y_train, model)
    initial_test_accuracy = accuracy(X_test, y_test, model)
    
    print(f"\nTraining model for {epochs} epochs...")
    train_losses, test_losses, train_accuracies, test_accuracies = train(
        model, loss_fn, lr, X_train, y_train, X_test, y_test, 
        epochs, batch_size, recorder
    )
    
    print("\nTraining complete!")
    print(f"Initial train loss: {initial_train_loss:.6f}, Initial test loss: {initial_test_loss:.6f}")
    print(f"Initial train accuracy: {initial_train_accuracy:.2%}, Initial test accuracy: {initial_test_accuracy:.2%}")
    print(f"Final train loss: {train_losses[-1]:.6f}, Final test loss: {test_losses[-1]:.6f}")
    print(f"Final train accuracy: {train_accuracies[-1]:.2%}, Final test accuracy: {test_accuracies[-1]:.2%}")
    
    # Generate video if recording was enabled
    if use_recording and recorder is not None:
        video_output_path = f'visualizations/decision_boundary_training_{dataset_type}.mp4'

        # Create output directory if needed
        output_dir = os.path.dirname(video_output_path) if os.path.dirname(video_output_path) else 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print("Generating decision boundary training video...")
        print("="*50)
        
        animate_decision_boundary_training(
            model, X_train, y_train, recorder,
            resolution=200, fps=10,
            save_path=video_output_path
        )
        
        print("\n" + "="*50)
        print(f"Video saved to {video_output_path}")
        print("="*50)
        print("\nNote: If MP4 file wasn't created, a GIF file should be available.")
        print("To create MP4 videos, install FFmpeg: brew install ffmpeg")


if __name__ == "__main__":
    main()
