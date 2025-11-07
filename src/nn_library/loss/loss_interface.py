class Loss:
  def __init__(self, y_real):
    raise NotImplementedError("Not implemented.")

  def calculate(self, y_pred):
    raise NotImplementedError("Not implemented.")

  def get_gradient(self):
    raise NotImplementedError("Not implemented.")