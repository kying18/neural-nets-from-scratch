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
    return 1 - self.fwd ** 2

  def step(self, lr):
    pass
