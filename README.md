# Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning

## 📑 Overview
This repository presents a proof-of-concept implementation of our proposed method, 
which enables adversaries to effectively remove backdoors from a centralized model 
in a Federated Learning framework. The adversary applies Machine Unlearning to eliminate 
backdoors once their objectives are achieved or when there is a suspicion of detection.

## 🖥️  Requirements
- **Python Version**: Python 3.12.4 (Confirmed compatibility).
- **CUDA Toolkit**: Cuda compilation tools (Tested with release 12.4, V12.4.131).

## 🛠️ Installation Guide
Set up a dedicated Python virtual environment and install required dependencies:
```bash
python -m venv fbu
source fbu/bin/activate
pip install -r requirements.txt
```

## 🗄️ Dataset and Model
- The project uses the `CIFAR-10` dataset for training and evaluation. The dataset 
  will be automatically downloaded into the `data` directory when running the code 
  for the first time.
- The project uses `VGG16` model for training on `CIFAR-10`. The model's architecture, along with
  other variants of the `VGG` network, is defined in `vgg_models.py` script.

## 🚀 Step-by-Step Execution Guide
- ### Basic Command Structure
    To run the training process, execute the `main.py` script. Below is the basic structure
    of the command:
    ```bash
    python main.py --num_rounds [NUMBER_OF_ROUNDS] \
               --poison \
               --unlearn \
               --poison_strategy [STRATEGY] \
               --poison_start_round [START_ROUND] \
               --poison_duration [DURATION] \
               --unlearn_duration [UNLEARN_DURATION]
    ```

- ### Available Arguments
    The script supports several arguments to configure and modify the training behavior:
    
    | Argument                                | Type   | Description                                                                   |
    |-----------------------------------------|--------|-------------------------------------------------------------------------------|
    | `--num_rounds [NUMBER_OF_ROUNDS]`       | `int`  | Number of rounds to run the federated learning process.                       |
    | `--poison`                              | `flag` | Enable backdoor poisoning during the training process.                        |
    | `--unlearn`                             | `flag` | Enable unlearning after the poisoning process.                                |
    | `--poison_strategy [STRATEGY]`          | `str`  | Strategy used for backdoor poisoning (e.g., "continuous", "fixed", "random"). |
    | `--poison_start_round [START_ROUND]`    | `int`  | The round number when backdoor poisoning should start.                        |
    | `--poison_duration [DURATION]`          | `int`  | Number of rounds for which the backdoor poisoning is active.                  |
    | `--unlearn_duration [UNLEARN_DURATION]` | `int`  | Number of rounds for which unlearning should be applied after poisoning.      |

- ### Execution Examples
  - #### Regular Training
      To execute standard federated learning without poisoning or unlearning,
      use the following command:
      ```bash
      python main.py --num_rounds [NUMBER_OF_ROUNDS]
      ```

  - #### Poisoning Mode
      To enable backdoor poisoning, use the `--poison` flag and configure `--poison_strategy`, 
      `--poison_start_round`, and `--poison_duration`.
      Use the following command for poisoning mode:
      ```bash
      python main.py --num_rounds [NUMBER_OF_ROUNDS] --poison --poison_strategy [STRATEGY] --poison_start_round [START_ROUND] --poison_duration [DURATION]
      ```

  - #### Poisoning Mode with Unlearning
      To enable unlearning for removing the backdoor effects after poisoning,
      use the `--unlearn` flag and configure the `--unlearn_duration`. Use the following
      command for poisoning mode with unlearning:
      ```bash
      python main.py --num_rounds [NUMBER_OF_ROUNDS] --poison --poison_strategy [STRATEGY] --poison_start_round [START_ROUND] --poison_duration [DURATION] --unlearn --unlearn_duration [UNLEARN_DURATION]
      ```
      The unlearning phase will begin after the poisoning phase is complete.

  - #### Configuration Parameters
      The default settings for this project can be found in the `config.py` file.
      Below is a brief overview of key configuration parameters.
      Adjust the parameters for more fine-grained control over the federated learning
      and poisoning behavior.
    
      | Configuration Category                  | Parameter         | Value | Description                                                  |
      |-----------------------------------------|-------------------|-------|--------------------------------------------------------------|
      | General Training Configuration          | `batch_size`      | 128   | Batch size used during training.                             |
      |                                         | `num_classes`     | 10    | Number of classes in the classification task.                |
      | Federated Learning Client Configuration | `num_clients`     | 100   | Total number of clients participating in federated learning. |
      |                                         | `num_selected`    | 10    | Number of clients selected per round for model updates.      |
      | Poisoning Attack Configuration          | `attacker_id`     | 42    | The ID of the client designated as the attacker.             |
      |                                         | `target_class`    | 0     | The target class for the backdoor attack.                    |
      |                                         | `num_poison`      | 250   | Number of poisoned samples introduced into the training.     |
      |                                         | `fixed_frequency` | 5     | Frequency at which attacker client is selected.              |
      | Unlearning Parameters                   | `alpha`           | 1     | Weight for the total training loss.                          |
      |                                         | `beta`            | 1     | Scaling factor for the penalty.                              |
      |                                         | `gamma`           | 3     | Weight for the backdoor loss.                                |

- ### Analysis of Results
    Each execution generates a log file named in the format `[FILE_NAME].log`.
    For example, regular training creates a log file named `round_history.log`, 
    while a poisoning mode training with a `random` poisoning strategy generates
    a log file named `round_history_backdoor_random.log`. Use `plot.py` to extract
    round-wise accuracy for clean test samples (`Test Accuracy`) and poisoned samples
    (`Backdoor Accuracy`) from the log file, and plot the results.
    The plot will be saved with the name `[FILE_NAME].png`.
    Use the following command to perform the analysis:
    ```bash
    python plot.py --file_name [FILE_NAME].log
    ```

- ### Reproducible Examples
    Three execution logs for regular training, poison mode of training, and poison mode of training with unlearning
    are provided in the `examples` directory. The configuration parameters used are:
    `[NUMBER_OF_ROUNDS]`=3000, `[STRATEGY]`=random, `[START_ROUND]`=300, `[DURATION]`=100, and `[UNLEARN_DURATION]`=30.

## 📚 Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam, Hithem Lamri, and Michail Maniatakos, "_Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning_", IEEE TAI 2024.

### BibTex Citation
```
@article{DBLP:journals/tai/AlamKM24,
  author       = {Manaar Alam and
                  Hithem Lamri and
                  Michail Maniatakos},
  title        = {{Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning}},
  journal      = {IEEE Transactions on Artificial Intelligence, {IEEE TAI} 2024},
  year         = {2024},
  url          = {https://doi.org/10.1109/TAI.2024.3465441},
  doi          = {10.1109/TAI.2024.3465441},
}
```

## 📩 Contact Us
For more information or help with the setup, please contact Manaar Alam at: [alam.manaar@nyu.edu](mailto:alam.manaar@nyu.edu)
