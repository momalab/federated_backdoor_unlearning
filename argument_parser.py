import argparse


def get_args():
    """
    Parses command-line arguments for the federated learning experiment.

    Returns:
        argparse.Namespace: Contains all the arguments passed via the command line,
                            with their default values if not provided.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Argument Parser")

    # Number of federated learning rounds to run
    parser.add_argument('--num_rounds', type=int, default=100,
                        help="Number of rounds to run the federated learning process.")

    # Flag to enable backdoor poisoning
    parser.add_argument('--poison', action='store_true',
                        help="Enable backdoor poisoning during the training process.")

    # Flag to enable unlearning after poisoning
    parser.add_argument('--unlearn', action='store_true',
                        help="Enable unlearning after the poisoning process.")

    # The strategy used for the backdoor poisoning
    parser.add_argument('--poison_strategy', type=str, default="normal",
                        help="Strategy used for backdoor poisoning.")

    # The round number to start applying the backdoor poisoning
    parser.add_argument('--poison_start_round', type=int, default=50,
                        help="The round number when backdoor poisoning should start.")

    # Duration of the poisoning phase (number of rounds)
    parser.add_argument('--poison_duration', type=int, default=50,
                        help="Number of rounds for which the backdoor poisoning is active.")

    # Duration of the unlearning phase (number of rounds)
    parser.add_argument('--unlearn_duration', type=int, default=10,
                        help="Number of rounds for which unlearning should be applied after poisoning.")

    # Parse and return the arguments
    return parser.parse_args()
