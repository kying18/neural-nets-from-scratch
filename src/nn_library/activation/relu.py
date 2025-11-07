import numpy as np

from ..models.model_interface import Model

class ReLU(Model):
  def __init__(self):
    pass

  def forward(self, X):
    self.curr_x = X
    return np.maximum(X, 0)

  def backward(self, dL_dout):
    mask = (self.curr_x > 0)
    # (batch, out_dim) * (batch, in_dim), where out_dim = in_dim for ReLU
    return dL_dout * mask

  def step(self, lr):
    pass
