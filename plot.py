import re
import matplotlib.pyplot as plt
import argparse
import os


def parse_log_file(file_name):
    """
    Parse a log file to extract test and backdoor accuracies over multiple rounds.

    Parameters:
        file_name (str): The path to the log file.

    Returns:
        tuple: A tuple containing three lists:
            - rounds: List of round numbers.
            - test_accuracies: List of test accuracies corresponding to each round.
            - backdoor_accuracies: List of backdoor accuracies corresponding to each round.
    """
    rounds = []
    test_accuracies = []
    backdoor_accuracies = []

    # Regular expression pattern to match the accuracy lines
    accuracy_pattern = re.compile(r'Test Accuracy:\s([\d.]+),\sBackdoor Accuracy\s->\s([\d.]+)')

    # Read the log file
    with open(file_name, 'r') as file:
        contents = file.readlines()

    current_round = None
    # Loop through each line in the log file to extract relevant data
    for line in contents:
        # Check for round number
        if 'Round:' in line:
            current_round = int(line.split()[-1])

        # Match accuracy lines using regex
        match = accuracy_pattern.search(line)
        if match:
            test_acc = float(match.group(1))
            backdoor_acc = float(match.group(2))
            rounds.append(current_round)
            test_accuracies.append(test_acc)
            backdoor_accuracies.append(backdoor_acc)

    return rounds, test_accuracies, backdoor_accuracies


def plot_accuracies(rounds, test_accuracies, backdoor_accuracies, output_file):
    """
    Plot the test and backdoor accuracies over the rounds and save the plot as a PNG image.

    Parameters:
        rounds (list): List of round numbers.
        test_accuracies (list): List of test accuracies corresponding to each round.
        backdoor_accuracies (list): List of backdoor accuracies corresponding to each round.
        output_file (str): The output file name where the plot will be saved.
    """
    # Plot settings
    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 18}
    colors = ["#00a896", "#6a0136"]

    # Plot the results
    plt.plot(rounds, test_accuracies, color=colors[0], label='Test Accuracy')
    plt.plot(rounds, backdoor_accuracies, color=colors[1], label='Backdoor Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend(prop={'family': 'serif', 'size': 14}, ncol=2, edgecolor='black', shadow=True, loc='upper center',
               bbox_to_anchor=(0.5, 1.15))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-5, 105)
    plt.xlabel("Federated Learning Rounds", fontdict=font)
    plt.ylabel("Accuracy (%)", fontdict=font)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function to parse the log file and plot the test and backdoor accuracies.
    It accepts the log file path as a command-line argument and saves the plot
    with the same base filename as the input, but with a `.png` extension.
    """
    # Set up argument parser to accept log file name from command line
    parser = argparse.ArgumentParser(description="Plot test and backdoor accuracies from a log file.")
    parser.add_argument('--file_name', type=str, help="The path to the log file to parse.")

    args = parser.parse_args()

    # Generate output file name by replacing the file extension with '.png'
    output_file = os.path.splitext(args.file_name)[0] + ".png"

    # Parse the log file and plot the results
    rounds, test_accuracies, backdoor_accuracies = parse_log_file(args.file_name)
    plot_accuracies(rounds, test_accuracies, backdoor_accuracies, output_file)


if __name__ == "__main__":
    main()
