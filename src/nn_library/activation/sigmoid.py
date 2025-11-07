import numpy as np

from ..models.model_interface import Model

class Sigmoid(Model):
  def __init__(self):
    pass

  def forward(self, X):
    exponential = np.exp(-X)
    self.fwd = 1 / (1 + exponential)
    return self.fwd

  def backward(self, dL_dout):
    return dL_dout * self.fwd * (1-self.fwd)

  def step(self, lr):
    pass
