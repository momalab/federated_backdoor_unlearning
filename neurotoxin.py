# Adapted from https://github.com/jhcknzzm/Federated-Learning-Backdoor

import torch
from torch.optim import SGD, lr_scheduler

import utils


def grad_mask_cv(model, train_loader, criterion, gradmask_ratio, device):
    """
        Computes a gradient mask by performing a forward and backward pass over the data
        to selectively freeze a portion of the model's gradients based on their magnitude.

        Args:
            model (nn.Module): The model being trained.
            train_loader (DataLoader): The data loader for the benign (non-poisoned) training data.
            criterion (nn.Module): The loss function.
            gradmask_ratio (float): The ratio of gradients to mask.
            device (torch.device): Device to run the computations.

        Returns:
            list: A list of gradient masks, one per layer of the model.
    """
    model.train()  # Set model to training mode
    model.zero_grad()  # Zero out gradients before starting

    # Perform a forward and backward pass to calculate gradients
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)

    mask_grad_list = []
    grad_list = []

    # Collect gradients from all layers and store their absolute values
    for _, params in model.named_parameters():
        if params.requires_grad:
            grad_list.append(params.grad.abs().view(-1))

    # Concatenate all gradient vectors into a single flat tensor
    grad_list = torch.cat(grad_list).to(device)

    # Select the top-k gradients to mask based on the gradmask_ratio
    _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * gradmask_ratio))
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
    mask_flat_all_layer[indices] = 1.0

    # Rebuild the mask for each layer based on the selected gradients
    count = 0
    for _, params in model.named_parameters():
        if params.requires_grad:
            gradients_length = len(params.grad.abs().view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length].to(device)
            mask_grad_list.append(mask_flat.reshape(params.grad.size()).to(device))
            count += gradients_length

    model.zero_grad()  # Zero out the gradients again
    return mask_grad_list


def apply_grad_mask(model, mask_grad_list):
    """
        Applies a gradient mask to the model's gradients, freezing certain portions of the model's weights.

        Args:
            model (nn.Module): The model being trained.
            mask_grad_list (list): The list of gradient masks to apply, one per layer of the model.
    """
    mask_grad_list_copy = iter(mask_grad_list)
    for name, params in model.named_parameters():
        if params.requires_grad:
            # Apply the mask by multiplying the gradients by the mask
            params.grad = params.grad * next(mask_grad_list_copy)


def poison(model, benign_loader, train_loader, client_index, test_loader, backdoor_loader, device, epochs=6, gradmask_ratio=0.75):
    """
        Trains a model with a backdoor poisoning attack by selectively applying a gradient mask
        to benign data, then conducting poisoned training with masked gradients.

        Args:
            model (nn.Module): The model being poisoned and trained.
            benign_loader (DataLoader): DataLoader for benign (non-poisoned) data.
            train_loader (DataLoader): DataLoader for poisoned training data.
            client_index (int): Index of the client being trained (for tracking).
            test_loader (DataLoader): DataLoader for normal test data.
            backdoor_loader (DataLoader): DataLoader for backdoor test data.
            device (torch.device): Device to run computations on.
            epochs (int, optional): Number of epochs to train. Default is 6.
            gradmask_ratio (float, optional): The ratio of gradients to mask. Default is 0.75.

        Returns:
            float: The accuracy of the poisoned model on test and backdoor data.
    """
    # Optimizer and learning rate scheduler setup
    optimizer = SGD(model.parameters(), lr=0.05)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 5], gamma=0.1, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    # Compute the gradient mask using benign data
    mask_grad_list = grad_mask_cv(model, benign_loader, criterion, gradmask_ratio, device)

    # Train the model for the specified number of epochs
    for e in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Apply gradient mask if necessary
            if gradmask_ratio != 1:
                apply_grad_mask(model, mask_grad_list)

            optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate

    # Evaluate the poisoned model on both normal and backdoor test data
    return utils.test_client(model, train_loader, client_index, test_loader, device, backdoor_loader)
