import numpy as np

from ..models.model_interface import Model

class Tanh(Model):
  def __init__(self):
    pass

  def forward(self, X):
    exponential = np.exp(X)
    neg_exp = np.exp(-X)

    self.fwd = (exponential - neg_exp) / (exponential + neg_exp)
    return self.fwd

  def backward(self, dL_dout):
    # Derivative of tanh is 1 - tanh(x)^2
    # Must multiply by incoming gradient for chain rule
    return dL_dout * (1 - self.fwd ** 2)

  def step(self, lr):
    pass
