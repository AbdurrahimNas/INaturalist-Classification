
import torchvision

from torchvision import transforms
from torch import nn

def create_model(num_classes:int):
  """
  Creates a pretrained MobileNet_V3_Large Feature Extractor model with default 
  weights.

  Keyword Arguments:
    :arg num_classes: Number of classes to classify.
    :type num_classes: int

  Example Usage:
    model, model_transform, test_transforms = create_model(num_classes=101)

  """
  model_weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
  model = torchvision.models.mobilenet_v3_large(weights=model_weights)
  test_transforms = model_weights.transforms()
  train_transforms = transforms.Compose([
      transforms.TrivialAugmentWide(num_magnitude_bins=31),
      test_transforms
  ])

  for param in model.parameters():
    param.requires_grad = True

  model.classifier = nn.Sequential(
      nn.Linear(in_features=960, out_features=1280),
      nn.Hardswish(),
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=num_classes)
  )

  return model, train_transforms, test_transforms

