import logging
import numpy as np

import torch
import torch.optim as optim

import config


def select_clients(selection, r):
    """
        Selects clients for a given federated learning round based on different strategies.

        Args:
            selection (str): The client selection strategy ('continuous', 'fixed', 'random', 'normal').
            r (int): The current round number.

        Returns:
            list: A list of client indices selected for the current round.
    """
    if selection == "continuous":
        # Always include the attacker and randomly select the remaining clients
        client_idx = [config.attacker_id]
        remaining_clients = np.setdiff1d(np.arange(config.num_clients), client_idx)
        additional_clients = np.random.choice(remaining_clients, config.num_selected - 1, replace=False).tolist()
        client_idx.extend(additional_clients)

    elif selection == "fixed":
        # Select the attacker every `fixed_frequency` rounds, otherwise select randomly
        if r % config.fixed_frequency == 0:
            client_idx = [config.attacker_id]
            remaining_clients = np.setdiff1d(np.arange(config.num_clients), client_idx)
            additional_clients = np.random.choice(remaining_clients, config.num_selected - 1, replace=False).tolist()
            client_idx.extend(additional_clients)
        else:
            remaining_clients = np.setdiff1d(np.arange(config.num_clients), [config.attacker_id])
            client_idx = np.random.choice(remaining_clients, config.num_selected, replace=False).tolist()

    elif selection == "random":
        # Randomly select clients with no specific preference
        client_idx = np.random.choice(np.arange(config.num_clients), config.num_selected, replace=False).tolist()

    elif selection == "normal":
        # Select clients excluding the attacker
        client_idx = np.random.choice(np.setdiff1d(np.arange(config.num_clients), [config.attacker_id]),
                                      config.num_selected, replace=False).tolist()

    return client_idx


def client_update(model, train_loader, client_index, test_loader, device, epochs=2):
    """
        Updates the model for a specific client by training it on the client's data.

        Args:
            model (nn.Module): The model to be updated.
            train_loader (DataLoader): DataLoader for the client's training data.
            client_index (int): The index of the client.
            test_loader (DataLoader): DataLoader for the test data.
            device (torch.device): Device for computations.
            epochs (int, optional): Number of epochs to train the model. Default is 2.

        Returns:
            float: Accuracy of the model on the test set after training.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic gradient descent optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    model.train()  # Set model to training mode

    # Train the model for the specified number of epochs
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    # Return the model's performance after training
    return test_client(model, train_loader, client_index, test_loader, device)


def test_client(model, train_loader, client_index, test_loader, device, backdoor_loader=None):
    """
        Tests the client's model on both training and test data, optionally testing on backdoor data.

        Args:
            model (nn.Module): The model being tested.
            train_loader (DataLoader): DataLoader for the client's training data.
            client_index (int): The index of the client.
            test_loader (DataLoader): DataLoader for the test data.
            device (torch.device): Device for computations.
            backdoor_loader (DataLoader, optional): DataLoader for backdoor test data. Default is None.

        Returns:
            None: Logs the accuracy on training, test, and backdoor data (if available).
    """

    # Calculate accuracy on training and test data
    train_acc = calculate_accuracy(model, train_loader, device)
    test_acc = calculate_accuracy(model, test_loader, device)

    if backdoor_loader is None:
        # Log and print results for non-backdoor testing
        logging.info(f'Client {client_index}: train acc -> {train_acc:.3f}, test acc -> {test_acc:.3f}')
        print(f'Client {client_index}: train acc -> {train_acc:.3f}, test acc -> {test_acc:.3f}')
    else:
        # Calculate and log backdoor accuracy if backdoor_loader is provided
        backdoor_acc = calculate_accuracy(model, backdoor_loader, device)
        logging.info(f'Client {client_index}: train acc -> {train_acc:.3f}, test acc -> {test_acc:.3f}, '
                     f'backdoor acc -> {backdoor_acc:.3f}')
        print(f'Client {client_index}: train acc -> {train_acc:.3f}, test acc -> {test_acc:.3f}, '
              f'backdoor acc -> {backdoor_acc:.3f}')


def calculate_accuracy(model, data_loader, device):
    """
        Calculates the accuracy of the model on a given dataset.

        Args:
            model (nn.Module): The model being tested.
            data_loader (DataLoader): DataLoader for the dataset to test on.
            device (torch.device): Device for computations.

        Returns:
            float: The accuracy of the model on the provided dataset.
    """
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)  # Get the index of the max log-probability
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total  # Return accuracy as a percentage


def server_aggregate(global_model, client_models):
    """
        Aggregates the parameters of client models to update the global model in federated learning.

        Args:
            global_model (nn.Module): The global model to be updated.
            client_models (list of nn.Module): The list of client models to aggregate.

        Returns:
            None: Updates the global model with the aggregated parameters from client models.
    """
    global_dict = global_model.state_dict()  # Get the global model's state dictionary

    # Compute the average of client models' parameters for each layer
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_model.state_dict()[k].float()
                                      for client_model in client_models], 0).mean(0)

    # Load the averaged parameters into the global model
    global_model.load_state_dict(global_dict)

    # Synchronize client models with the updated global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
