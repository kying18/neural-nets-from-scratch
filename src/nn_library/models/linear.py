import numpy as np

from model_interface import Model

class Linear(Model):
  def __init__(self, in_dim, out_dim):
    # Better initialization: small random values centered around zero
    # Using Xavier/Glorot initialization for better convergence
    scale = np.sqrt(2.0 / (in_dim + out_dim))
    self.W = np.random.randn(in_dim, out_dim) * scale
    self.b = np.zeros(out_dim)

  # X is a tensor (batch, in_dim)
  def forward(self, X):
    # (batch, in_dim) * (in_dim, out_dim)
    self.curr_x = X
    self.predictions = X @ self.W + self.b
    return self.predictions

  # dL_dout is the gradient with respect to the y_pred output
  def backward(self, dL_dout):
    # (batch, in_dim).T @ (batch, out_dim)
    self.dL_dw = self.curr_x.T @ dL_dout
    self.dL_db = np.sum(dL_dout, axis=0)
    # (batch, out_dim) * (in_dim, out_dim).T
    dL_dx = dL_dout @ self.W.T
    return dL_dx

  def step(self, lr):
    self.W = self.W - lr * self.dL_dw
    self.b = self.b - lr * self.dL_db
