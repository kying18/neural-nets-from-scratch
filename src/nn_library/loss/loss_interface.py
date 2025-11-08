class Loss:
  def __init__(self):
    """Initialize loss function. No parameters needed - y_real is passed to calculate()."""
    pass

  def calculate(self, y_pred, y_real):
    """
    Calculate loss given predictions and true labels.
    
    Args:
      y_pred: Model predictions
      y_real: True labels
      
    Returns:
      Scalar loss value
    """
    raise NotImplementedError("Not implemented.")

  def get_gradient(self):
    """
    Get gradient of loss with respect to predictions.
    Must be called after calculate().
    
    Returns:
      Gradient dL/dy_pred
    """
    raise NotImplementedError("Not implemented.")