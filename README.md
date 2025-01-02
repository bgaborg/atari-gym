# Atari Gym Agent

This project implements a reinforcement learning agent for playing Atari games using the Arcade Learning Environment (ALE) and PyTorch.

## Project Structure

- `agent.py`: Contains the implementation of the reinforcement learning agent.
- `neural.py`: Defines the neural network architecture used by the agent.
- `main.py`: The main script to run the training and evaluation of the agent.
- `wrappers.py`: Custom environment wrappers to modify the behavior of the environment.

## Requirements

- Python 3.11
- PyTorch
- Gymnasium
- Arcade Learning Environment (ALE)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/atari-gym-agent.git
    cd atari-gym-agent
    ```

2. Create a virtual environment with uv, and install the required packages:
    ```sh
    uv venv
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    uv pip install -r pyproject.toml
    ```

## Usage

1. To train the agent, run the [main.py](http://_vscodecontentref_/1) script:
    ```sh
    python main.py
    ```

2. To evaluate the agent, modify the [main.py](http://_vscodecontentref_/2) script to load a pre-trained model and run the evaluation loop.

3. Modify the `my_rig_factor` parameter in the [agent.py](http://_vscodecontentref_/3) script to adjust the memory replay buffer size.

4. Install pytorch with cuda support if cuda is available on your machine:
    ```sh
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

## Custom Wrappers

### SkipFrame

The `SkipFrame` wrapper returns only every `skip`-th frame and accumulates rewards over skipped frames.

### ActionSpaceWrapper

The `ActionSpaceWrapper` restricts the action space to a subset of allowed actions.

## Troubleshooting
If you encounter any issues, please check the following:

Ensure that all dependencies are installed correctly.
- Verify that your environment is set up correctly (e.g., virtual environment is activated).
- Check the device (CPU/GPU) compatibility and ensure that the data is correctly moved between devices if necessary.

# Contributing
Contributions are welcome! Please open an issue or submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

