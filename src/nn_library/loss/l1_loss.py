import numpy as np

from loss_interface import Loss

class L2Loss(Loss):
  def __init__(self, y_real):
    self.y_real = y_real
    self.batch_size = self.y_real.shape[0]

  def calculate(self, y_pred):
    self.y_pred = y_pred
    abs_errors = np.abs(self.y_pred - self.y_real)
    return np.mean(abs_errors)

  # returns gradient dL/dy_pred
  def get_gradient(self):
    pos = (self.y_pred - self.y_real) > 0
    return (pos * 2 - 1) / self.batch_size