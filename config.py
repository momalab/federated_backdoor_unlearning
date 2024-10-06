# Configuration for the federated learning experiment

# General training configuration
batch_size = 128  # Batch size used during training
num_classes = 10  # Number of classes in the classification task

# Federated learning client configuration
num_clients = 100  # Total number of clients participating in federated learning
num_selected = 10  # Number of clients selected per round for model updates

# Poisoning attack configuration
attacker_id = 42  # The ID of the client designated as the attacker
target_class = 0  # The target class for the backdoor attack
num_poison = 250  # Number of poisoned samples introduced into the training process
fixed_frequency = 5  # Frequency at which attacker client is selected

# Unlearning parameters
alpha = 1  # Weight for the total training loss
beta = 1  # Scaling factor for the penalty
gamma = 3  # Weight for the backdoor loss
