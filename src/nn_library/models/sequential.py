from model_interface import Model

class Sequential(Model):
  def __init__(self, layers):
      self.layers = layers

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)

    return X
  
  def backward(self, dL_dout):
    for layer in self.layers[::-1]:
      dL_dout = layer.backward(dL_dout)

  def step(self, lr):
    for layer in self.layers:
      layer.step(lr)
