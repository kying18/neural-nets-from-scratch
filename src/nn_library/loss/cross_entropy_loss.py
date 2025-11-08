import numpy as np

from .loss_interface import Loss

_EPS = 1e-12

class BinaryCrossEntropyLoss(Loss):
  def __init__(self):
    super().__init__()

  def calculate(self, y_pred, y_real):
    self.y_pred = y_pred
    self.y_real = y_real
    self.batch_size = y_real.shape[0]
    # y_pred are probabilistic outputs of length N
    # -(y*log(p) + (1-y)*log(1-p))
    clipped = np.clip(y_pred, _EPS, 1-_EPS)
    cross_entropy = -(self.y_real * np.log(clipped) + (1 - self.y_real) * np.log(1 - clipped))
    return np.mean(cross_entropy)

  def get_gradient(self):
    gradient = (1 - self.y_real) / (1 - self.y_pred) - self.y_real / self.y_pred
    return gradient / self.batch_size


class CategoricalCrossEntropyLoss(Loss):
  def __init__(self):
    super().__init__()

  def calculate(self, y_pred, y_real):
    self.y_pred = y_pred
    self.y_real = y_real
    self.batch_size = y_real.shape[0]
    # sum of y_i * log(p_i)
    # y_pred is (N, # of classes), where prob of each class is the output
    clipped = np.clip(y_pred, _EPS, 1 - _EPS)
    # y_real is one-hot encoded, so also (N, # of classes)
    cross_entropy = -np.sum(self.y_real * np.log(clipped), axis=1)
    return np.mean(cross_entropy)

  def get_gradient(self):
    gradient = - self.y_real / self.y_pred
    return gradient / self.batch_size