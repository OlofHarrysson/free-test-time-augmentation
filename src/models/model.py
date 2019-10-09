import torch
import torch.nn as nn
import torchvision.models as models


def get_model(config):
  model = MyModel(config)
  model = model.to(model.device)
  return model


class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.loss_fn = nn.CrossEntropyLoss()

    self.backbone = models.resnet18(pretrained=config.pretrained)
    n_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Linear(n_features, 10)

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    return self.backbone(x)

  def calc_loss(self, outputs, labels, accuracy=False):
    labels = labels.to(self.device)

    if accuracy:
      _, preds = torch.max(outputs, 1)
      accuracy = torch.sum(preds == labels, dtype=float) / len(preds)
      return self.loss_fn(outputs, labels), accuracy
    return self.loss_fn(outputs, labels)

  def predict(self, inputs, test_time_aug=True):
    with torch.no_grad():
      if test_time_aug:
        return self.test_time_aug(inputs)
      return self(inputs)

  def test_time_aug(self, x):
    x = x.to(self.device)

    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)

    x = self.backbone.layer1(x)
    x = self.backbone.layer2(x)
    x = self.backbone.layer3(x)
    x = self.backbone.layer4(x)

    x = x.view(x.shape[0], x.shape[1], 1, -1)
    all_xs = 0
    for cell_i in range(x.shape[-1]):
      mask = torch.ones(x.shape, dtype=torch.bool, device=x.device)
      mask[..., cell_i] = 0
      xs = torch.masked_select(x, mask)
      xs = xs.view(x.shape[0], x.shape[1], 1, -1)
      xs = self.backbone.avgpool(xs)
      xs = torch.flatten(xs, 1)
      xs = self.backbone.fc(xs)
      xs = self.softmax(xs)
      all_xs += xs

    x = self.softmax(all_xs)
    return x

    # x = self.backbone.avgpool(x)
    # print(x.shape)

    # qwe

    # x2 = x[:, :, 0:2, 0:2]
    # x3 = x[:, :, 0:2, 1:3]
    # x4 = x[:, :, 1:3, 0:2]
    # x5 = x[:, :, 1:3, 1:3]

    # x = self.mini_pred(x)
    # x2 = self.mini_pred(x2)
    # x3 = self.mini_pred(x3)
    # x4 = self.mini_pred(x4)
    # x5 = self.mini_pred(x5)

    # x = x + x2 + x3 + x4 + x5
    # x = self.softmax(x)

    # return x

  def mini_pred(self, x):
    x = self.backbone.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.backbone.fc(x)
    return self.softmax(x)


if __name__ == '__main__':
  import types
  config = types.SimpleNamespace()
  config.pretrained = False
  config.use_gpu = False
  model = MyModel(config)

  model.predict(torch.randn(8, 3, 96, 96))
