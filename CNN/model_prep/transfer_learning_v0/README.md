# CustomResnet101

`CustomResnet101` is a custom implementation of the ResNet-101 architecture with a modified fully connected (FC) layer at the end. It is designed for computer vision tasks, such as image classification.

## Usage

Instantiate the `CustomResnet101` model by specifying the number of classes for your custom classifier:

```python
model = CustomResnet101(num_classes=10)
```

Class: CustomResnet101

Description:
This class represents a custom implementation of the ResNet-101 architecture with a modified fully connected (FC) layer at the end. It is a subclass of the `nn.Module` class from PyTorch and can be used for various computer vision tasks, such as image classification.

Constructor:
CustomResnet101(num_classes)

Parameters:

- num_classes (int): The number of classes for the custom classifier. It determines the output size of the modified FC layer.

Methods:

- **init**(num_classes): Initializes an instance of the CustomResnet101 class.
- forward(x): Performs the forward pass of the model.

Attributes:

- resnet (ResNet): The ResNet-101 model loaded from torchvision.models, with pre-trained weights from ImageNet.
- fc_in_features (int): The number of input features to the modified FC layer.

Usage:

1. Instantiate the CustomResnet101 model:
   model = CustomResnet101(num_classes=10)

2. Forward pass:
   output = model(x)

Note:

- This implementation assumes that the ResNet-101 architecture has been defined and implemented separately within the class. The code provided does not include the actual architecture definition.
- It is recommended to define the architecture layers, blocks, and connections within the `__init__()` method of the CustomResnet101 class.
- The `super()` function is used to call the constructor of the parent class (`nn.Module`) and initialize the model.
- The `resnet101` function from `torchvision.models` is used to load the ResNet-101 architecture with pre-trained weights from the ImageNet dataset.
- The modified FC layer is defined as a sequential container with a single `nn.Linear` module, where the input size is `fc_in_features` (number of input features) and the output size is `num_classes` (number of classes).
- The `forward()` method performs the forward pass by passing the input `x` through the ResNet-101 architecture and returning the output.
