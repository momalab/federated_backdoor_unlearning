# Adapted from https://github.com/kuangliu/pytorch-cifar

import torch.nn as nn

# Configuration dictionary for different VGG variants (number of layers)
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
        A PyTorch implementation of the VGG model architecture. Supports VGG11, VGG13, VGG16, and VGG19.

        Args:
            vgg_name (str): The name of the VGG configuration (e.g., 'VGG16').
            num_classes (int): The number of output classes for classification.
            channels (int): The number of input channels (e.g., 3 for RGB images).
    """
    def __init__(self, vgg_name, num_classes, channels):
        super(VGG, self).__init__()
        self.in_channels = channels

        # Create the feature extraction layers based on the VGG configuration
        self.features = self._make_layers(cfg[vgg_name])

        # Define the fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # Fully connected layer 1
            nn.ReLU(True),  # ReLU activation
            nn.Linear(512, 512),  # Fully connected layer 2
            nn.ReLU(True),  # ReLU activation
            nn.Linear(512, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        """
            Forward pass through the VGG model.

            Args:
                x (Tensor): Input tensor representing the batch of images.

            Returns:
                Tensor: Output predictions from the classifier.
        """
        out = self.features(x)  # Extract features using convolutional layers
        out = out.view(out.size(0), -1)  # Flatten the output for the classifier
        out = self.classifier(out)  # Pass through fully connected layers
        return out

    def _make_layers(self, cfg):
        """
            Creates the convolutional layers and pooling layers based on the VGG configuration.

            Args:
                cfg (list): The configuration list specifying the number of filters and 'M' for max pooling.

            Returns:
                nn.Sequential: A sequential container with the constructed layers.
        """
        layers = []
        in_channels = self.in_channels

        # Iterate over the configuration list to build the layers
        for x in cfg:
            if x == 'M':
                # Max pooling layer
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Convolutional layer, followed by batch normalization and ReLU activation
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x  # Update the number of input channels for the next layer

        # Add a final average pooling layer
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
