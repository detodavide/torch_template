# CustomCNNDataset

This class represents a custom dataset for a convolutional neural network (CNN). It is designed to load and preprocess image data from a specified directory. The dataset can be used for tasks such as image classification.

## Installation

1. Clone the repository:
   Copy
   git clone https://github.com/detodavide/torch_template.git

2. Navigate to the `CNN/data_prep` directory:
   Copy
   cd torch_template/CNN/data_prep

3. Import the class:

```python
from custom_cnn_dataset import CustomCNNDataset
```

Copy
Usage
Initialization
To initialize the dataset, create an instance of the CustomCNNDataset class:

dataset = CustomCNNDataset(root_dir='path/to/data', transform=data_transform)
Copy
The root_dir parameter specifies the root directory containing the image data. It should have subdirectories, where each subdirectory represents a different class of images. The transform parameter is an optional function or transformation to apply to the loaded images. It can be used for data augmentation or preprocessing.

Length
To access the length of the dataset, use the len() method:

length = len(dataset)
Copy
Retrieval
To retrieve a specific image and its corresponding label from the dataset, based on the given index, use the getitem() method:

image, label = dataset[0]
Copy
Splitting
To split the dataset into training and validation subsets, use the splitter() method:

train_split, val_split = dataset.splitter(splits=[85, 15])
Copy
The splits parameter is a list specifying the desired split ratios for training and validation subsets. The sum of the ratios should be equal to 100. The default value is [85, 15].

Attributes
The following attributes are available for the CustomCNNDataset class:

root_dir: The root directory containing the image data.
transform: The transformation function to be applied to the loaded images.
classes: A sorted list of class names, derived from the subdirectories in the root directory.
image_paths: A list of paths to all the images in the dataset

```

```
