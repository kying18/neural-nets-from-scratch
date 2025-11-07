import numpy as np

from ..models.model_interface import Model

class Softmax(Model):
  def __init__(self):
    pass

  def forward(self, X):
    exponential = np.exp(X - np.max(X, axis=1, keepdims=True))
    self.fwd = exponential / np.sum(exponential, axis=1, keepdims=True)
    return self.fwd

  def backward(self, dL_dout):
    # dSi/dzj = Si * (1 * [i == j] - Sj) = J
    # dL/dzj = J@g = s*(g - sTg)
    dot_prod = np.sum(self.fwd * dL_dout, axis=1, keepdims=True)
    return self.fwd * (dL_dout - dot_prod)

  def step(self, lr):
    pass
