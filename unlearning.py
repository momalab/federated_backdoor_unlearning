import torch
import torch.optim as optim

import config
import utils


def unlearn(model, train_loader, client_index, test_loader, train_backdoor_loader, test_backdoor_loader, unlearn_count,
            device, unlearn_epoch=6):
    """
        Implements the unlearning process to mitigate the effects of backdoor attacks on a client model.
        The unlearning combines clean data training with backdoor-specific penalties.

        Args:
            model (nn.Module): The model to be unlearned (updated).
            train_loader (DataLoader): DataLoader for clean training data (non-poisoned).
            client_index (int): The index of the client whose model is being unlearned.
            test_loader (DataLoader): DataLoader for clean test data (non-poisoned).
            train_backdoor_loader (DataLoader): DataLoader for backdoor-poisoned training data.
            test_backdoor_loader (DataLoader): DataLoader for backdoor-poisoned test data.
            unlearn_count (int): Count of how many unlearning rounds have been applied.
            device (torch.device): Device for computations.
            unlearn_epoch (int, optional): Number of epochs for the unlearning process. Default is 6.

        Returns:
            float: Accuracy of the unlearned model on test and backdoor test data after unlearning.
    """
    # Adjust learning rate based on unlearn_count
    learning_rate = 5e-5 / (2 ** (unlearn_count / 10))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Save original parameters for calculating the unlearning penalty
    original_params = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            original_params.append(p)

    model.train()

    # Start the unlearning epochs
    for epoch in range(unlearn_epoch):
        params = []
        for _, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)

        clean_loss = 0  # Loss on clean (non-poisoned) data
        backdoor_loss = 0  # Negative loss on backdoor (poisoned) data

        # Compute the loss on clean data
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            clean_loss += criterion(outputs, targets)

        # Compute the loss on backdoor data
        for inputs, targets in train_backdoor_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            backdoor_loss -= criterion(outputs, targets)  # Backdoor loss is subtracted

        # Calculate total loss with backdoor mitigation
        total_loss = clean_loss + config.gamma * backdoor_loss

        # Calculate importance of clean data gradients
        clean_importance = []
        clean_loss.backward(retain_graph=True)
        for _, p in model.named_parameters():
            if p.requires_grad:
                clean_importance.append(p.grad)

        # Calculate importance of backdoor data gradients
        backdoor_importance = []
        backdoor_loss.backward(retain_graph=True)
        for _, p in model.named_parameters():
            if p.requires_grad:
                backdoor_importance.append(p.grad)

        # Compute penalty based on clean and backdoor importance
        penalty = 0
        for i in range(len(params)):
            importance = torch.nan_to_num(torch.div(clean_importance[i], backdoor_importance[i]), 1e-12)
            penalty += torch.norm(importance * torch.abs(params[i] - original_params[i]), 1)

        # Combine the total loss and penalty to compute the final unlearning loss
        unlearn_loss = config.alpha * total_loss + config.beta * penalty
        unlearn_loss.backward()
        optimizer.step()

    # Test the model after unlearning on both clean and backdoor data
    return utils.test_client(model, train_loader, client_index, test_loader, device, test_backdoor_loader)
