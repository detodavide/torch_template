Class: CustomCNNDataset

Description:
This class represents a custom dataset for a convolutional neural network (CNN). It is designed to load and preprocess image data from a specified directory. The dataset can be used for tasks such as image classification.

Constructor:
CustomCNNDataset(root_dir, transform=None)

Parameters:

- root_dir (string): The root directory containing the image data. It should have subdirectories, where each subdirectory represents a different class of images.
- transform (callable, optional): A function or transformation to apply to the loaded images. It can be used for data augmentation or preprocessing. Default is None.

Methods:

- **len**(): Returns the total number of images in the dataset.
- **getitem**(index): Retrieves a specific image and its corresponding label from the dataset, based on the given index.
- splitter(splits=[85,15]): Splits the dataset into training and validation subsets.

  - Parameters:

    - splits (list, optional): A list specifying the desired split ratios for training and validation subsets. The sum of the ratios should be equal to 100. Default is [85, 15].

  - Returns:
    - train_split (Subset): A subset of the dataset containing the training images and their labels.
    - val_split (Subset): A subset of the dataset containing the validation images and their labels.

Attributes:

- root_dir (string): The root directory containing the image data.
- transform (callable): The transformation function to be applied to the loaded images.
- classes (list): A sorted list of class names, derived from the subdirectories in the root directory.
- image_paths (list): A list of paths to all the images in the dataset.
- labels (list): A list of labels corresponding to each image in the dataset.

Usage:

1. Initialize the dataset:
   dataset = CustomCNNDataset(root_dir='path/to/data', transform=data_transform)

2. Access the length of the dataset:
   length = len(dataset)

3. Retrieve a specific image and its label:
   image, label = dataset[index]

4. Split the dataset into training and validation subsets:
   train_set, val_set = dataset.splitter(splits=[80, 20])

Note:

- It is assumed that the directory structure follows the convention of having subdirectories named after the classes and containing the corresponding images.
- The dataset assumes that the images are in RGB format. If the images are in a different format, modifications to the code may be required.
- The one-hot encoding of the labels is performed using the `one_hot` function, which converts the label to a one-hot tensor representation.
- The `splitter` method uses the `random_split` function from PyTorch to split the dataset randomly based on the given split ratios.
