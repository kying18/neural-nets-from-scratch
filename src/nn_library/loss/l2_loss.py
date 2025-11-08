import numpy as np

from .loss_interface import Loss

class L2Loss(Loss):
  def __init__(self):
    super().__init__()

  def calculate(self, y_pred, y_real):
    self.y_pred = y_pred
    self.y_real = y_real
    self.batch_size = self.y_real.shape[0]
    squared_errors = np.square(self.y_pred - self.y_real)
    return np.mean(squared_errors)

  # returns gradient dL/dy_pred
  def get_gradient(self):
    return 2 / self.batch_size * (self.y_pred - self.y_real)