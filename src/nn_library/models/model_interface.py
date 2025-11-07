class Model:
  def forward(self, X):
    raise NotImplementedError("Not implemented")

  def backward(self, dL_dout):
    raise NotImplementedError("Not implemented")

  def step(self, lr):
    raise NotImplementedError("Not implemented")