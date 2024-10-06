import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

import config


def apply_backdoor_pattern(img):
    """
        Applies a backdoor pattern to the input image by modifying certain pixels.

        Args:
            img (Tensor): The input image tensor to which the backdoor pattern will be applied.

        Returns:
            Tensor: The image tensor with the backdoor pattern applied.
    """
    # Set specific pixel locations to 0 to create a visible backdoor pattern
    img[:, 2, 2] = 0
    img[:, 3, 3] = 0
    img[:, 4, 4] = 0
    img[:, 4, 2] = 0
    img[:, 2, 4] = 0
    return img


def get_dataset():
    """
        Loads the CIFAR-10 dataset and splits the training data into `num_clients` partitions for federated learning.

        Returns:
            tuple: A tuple containing:
                - train_splits (list of Subset): A list of datasets for each client.
                - testdata (Dataset): The testing dataset.
    """
    transform = transforms.ToTensor()

    # Load CIFAR-10 dataset
    traindata = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testdata = datasets.CIFAR10('./data', train=False, transform=transform)

    # Split training data into subsets, one for each client
    split_size = traindata.data.shape[0] // config.num_clients
    train_splits = random_split(traindata, [split_size] * config.num_clients)

    return train_splits, testdata


def create_poisoned_data(dataset, target_class, num_poison=None):
    """
        Creates a dataset of poisoned images where a backdoor pattern is applied and all labels are set to the target class.

        Args:
            dataset (Dataset): The input dataset to poison.
            target_class (int): The class label assigned to the poisoned images.
            num_poison (int, optional): The number of poisoned samples to create. If None, poisons the entire dataset.

        Returns:
            TensorDataset: A dataset containing poisoned images and their corresponding target labels.
    """
    backdoor_images = []
    count = 0

    # Apply backdoor pattern to the dataset and set labels to the target class
    for img, _ in dataset:
        img = apply_backdoor_pattern(img)
        backdoor_images.append(img)
        count += 1
        if num_poison is not None and count == num_poison:
            break

    # Stack images into a single tensor and create corresponding labels
    backdoor_images = torch.stack(backdoor_images)
    backdoor_labels = torch.full((len(backdoor_images),), target_class, dtype=torch.int64)

    return TensorDataset(backdoor_images, backdoor_labels)


def get_poison_data(train_splits, testdata):
    """
        Creates loaders for benign and poisoned data, combining backdoor poisoned samples with benign samples for training.

        Args:
            train_splits (list of Subset): A list of datasets for each client.
            testdata (Dataset): The test dataset.

        Returns:
            tuple: Contains:
                - benign_loader (DataLoader): DataLoader for benign training data from the attacker client.
                - mixed_loader (DataLoader): DataLoader for a mix of benign and poisoned data.
                - poison_train_loader (DataLoader): DataLoader for training data with only poisoned samples.
                - poison_test_loader (DataLoader): DataLoader for testing data with only poisoned samples.
    """
    # Select the attacker's dataset from the client partitions
    attacker_dataset = train_splits[config.attacker_id]

    # Create a dataset with poisoned images (for the attacker)
    poison_train_dataset = create_poisoned_data(attacker_dataset, config.target_class, num_poison=config.num_poison)

    # Mix benign images with poisoned images for training
    mixed_images, mixed_labels = [], []
    for img, label in attacker_dataset:
        mixed_images.append(img)
        mixed_labels.append(torch.tensor(label))

    # Add poisoned samples to the mixed dataset
    for img, label in poison_train_dataset:
        mixed_images.append(img)
        mixed_labels.append(label)

    mixed_dataset = TensorDataset(torch.stack(mixed_images), torch.stack(mixed_labels))

    # Create poisoned test dataset
    poison_test_dataset = create_poisoned_data(testdata, config.target_class)

    # Create DataLoaders for different types of datasets
    benign_loader = DataLoader(attacker_dataset, batch_size=config.batch_size, shuffle=False)
    mixed_loader = DataLoader(mixed_dataset, batch_size=config.batch_size, shuffle=True)
    poison_train_loader = DataLoader(poison_train_dataset, batch_size=config.batch_size, shuffle=False)
    poison_test_loader = DataLoader(poison_test_dataset, batch_size=config.batch_size, shuffle=True)

    return benign_loader, mixed_loader, poison_train_loader, poison_test_loader
