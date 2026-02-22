# Environment Setup Guide

This project requires a Conda environment named `ros_env` with specific Python packages for robotics and reinforcement learning.

## Step 1: Create the Conda Environment

Open your terminal and run the following command to create a new environment with Python 3.11:

```bash
conda create -n ros_env python=3.11 -y
```

## Step 2: Activate the Environment

Activate the newly created environment:

```bash
conda activate ros_env
```

## Step 3: Install Requirements

Once the environment is active, install the necessary dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Required Packages

The following packages will be installed:

- **gymnasium** (v1.2.3): For reinforcement learning environments.
- **pybullet** (v3.25): For physics simulation.
- **numpy** (v2.4.2): For numerical computations.
- **stable-baselines3** (v2.7.1): For PPO and other RL algorithms.
- **torch** (v2.10.0): Deep learning backend for Stable-Baselines3.
- **matplotlib** (v3.10.6): For plotting and visualization.
- **pandas** (v3.0.0): For data manipulation and analysis.
- **Pillow** (v11.3.0): For image processing.
- **tqdm** (v4.67.3): For progress bars during training.

## Usage

In your Python scripts, you can now import these libraries:

```python
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time
```
