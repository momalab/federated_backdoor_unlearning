import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import config
import unlearning
import utils
import neurotoxin

from argument_parser import get_args
from data_loader import get_dataset, get_poison_data
from vgg_model import VGG


def main(args):
    """
        Main function to run the federated learning pipeline. It handles the entire learning process,
        including the poisoning and unlearning phases, based on the arguments provided.

        Args:
            args: Parsed command line arguments containing configurations for the training process.
    """
    # Determine device to use (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Append string to log filenames to differentiate between normal, backdoor, and unlearn phases
    append = ""
    if args.poison:
        if args.unlearn:
            append = f"_unlearn_{args.poison_strategy}"
        else:
            append = f"_backdoor_{args.poison_strategy}"

    # Configure logging
    logging.basicConfig(filename=f'round_history{append}.log', level=logging.INFO, format='%(message)s')

    # Load dataset
    train_splits, testdata = get_dataset()
    train_loader = [DataLoader(x, batch_size=config.batch_size, shuffle=False) for x in train_splits]
    test_loader = DataLoader(testdata, batch_size=config.batch_size, shuffle=False)

    # If poisoning is enabled, load poisoned data
    if args.poison:
        benign_loader, mixed_loader, poison_train_loader, poison_test_loader = get_poison_data(train_splits, testdata)

    # Initialize the global model and client models
    global_model = VGG('VGG16', num_classes=config.num_classes, channels=3).to(device)
    client_models = [VGG('VGG16', num_classes=config.num_classes, channels=3).to(device)
                     for _ in range(config.num_selected)]

    # Initialize counters and flags for backdoor and unlearn processes
    backdoor_count, unlearn_count = 0, 0
    backdoor_done = False
    unlearn_start = 0

    # Start the federated learning rounds
    for r in range(args.num_rounds):
        print(f'Round: {r + 1}')
        logging.info('----------------------')
        logging.info(f'Round: {r + 1}')
        logging.info('----------------------')

        # Determine the current learning phase (normal, backdoor, unlearn)
        learning_flag = "normal"
        if args.poison:
            if r >= args.poison_start_round and backdoor_count < args.poison_duration:
                learning_flag = "backdoor"
        if args.unlearn and backdoor_done:
            if r >= unlearn_start and unlearn_count < args.unlearn_duration:
                learning_flag = "unlearn"

        # Federated Learning - Normal Phase
        if learning_flag == "normal":
            # Select clients for normal update
            client_idx = utils.select_clients(learning_flag, r)
            logging.info('Selected Clients for Update: ' + str(client_idx))
            print('Selected Clients for Update: ' + str(client_idx))

            # Perform client update for selected clients
            for i in tqdm(range(config.num_selected)):
                utils.client_update(
                    client_models[i], train_loader[client_idx[i]], client_idx[i], test_loader, device
                )

        # Federated Learning - Backdoor (Poisoning) Phase
        if learning_flag == "backdoor":
            # Select clients for backdoor update
            client_idx = utils.select_clients(args.poison_strategy, r)
            logging.info('Selected Clients for Update: ' + str(client_idx))
            print('Selected Clients for Update: ' + str(client_idx))

            # Perform normal update for non-attacker clients, and poison update for attacker client
            for i in tqdm(range(config.num_selected)):
                if client_idx[i] != config.attacker_id:
                    # Normal client update
                    utils.client_update(
                        client_models[i], train_loader[client_idx[i]], client_idx[i], test_loader, device
                    )
                else:
                    # Attacker client update
                    neurotoxin.poison(
                        client_models[i], benign_loader, mixed_loader, client_idx[i], test_loader, poison_test_loader, device
                    )
                    backdoor_count += 1
                    # Check if backdoor phase is completed
                    if backdoor_count == args.poison_duration:
                        backdoor_done = True
                        unlearn_start = r + 1

        # Federated Learning - Unlearning Phase
        if learning_flag == "unlearn":
            # Select clients for unlearning update
            client_idx = utils.select_clients(args.poison_strategy, r)
            logging.info('Selected Clients for Update: ' + str(client_idx))
            print('Selected Clients for Update: ' + str(client_idx))

            # Perform normal update for non-attacker clients, and unlearn update for attacker clients
            for i in tqdm(range(config.num_selected)):
                if client_idx[i] != config.attacker_id:
                    # Normal client update
                    utils.client_update(
                        client_models[i], train_loader[client_idx[i]], client_idx[i], test_loader, device
                    )
                else:
                    # Perform unlearning on the attacker client
                    unlearning.unlearn(
                        client_models[i], train_loader[client_idx[i]], client_idx[i], test_loader, poison_train_loader, poison_test_loader, unlearn_count, device
                    )
                    unlearn_count += 1

        # Aggregate client models to update the global model
        utils.server_aggregate(global_model, client_models)

        # Evaluate the global model on test and poison data
        test_acc = utils.calculate_accuracy(global_model, test_loader, device)
        backdoor_acc = 0
        if args.poison:
            backdoor_acc = utils.calculate_accuracy(global_model, poison_test_loader, device)

        # Log and print the results for the current round
        logging.info(f'Test Accuracy: {test_acc:.3f}, Backdoor Accuracy -> {backdoor_acc:.3f}')
        print(f'Test Accuracy: {test_acc:.3f}, Backdoor Accuracy -> {backdoor_acc:.3f}')


if __name__ == '__main__':
    """
        Entry point of the program. It parses the command-line arguments and initiates the main training process.
    """
    arguments = get_args()
    main(arguments)
