import torch
import torch.nn as nn


class Validator():
  def __init__(self, config):
    pass

  def validate(self, model, val_loader, step, test_time_aug=True):
    n_corrects = 0
    n_seen = 0

    for batch_i, data in enumerate(val_loader, 1):
      inputs, labels = data
      outputs = model.predict(inputs, test_time_aug)
      _, preds = torch.max(outputs, 1)
      n_corrects += torch.sum(preds.cpu() == labels, dtype=float).item()
      n_seen += preds.shape[0]

    acc = n_corrects / n_seen
    print(f'Accuracy: {acc}\tTest time aug:{test_time_aug}')
