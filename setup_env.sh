#!/bin/bash

# filepath: /Users/rgower/Code/matsign/setup_env.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Create a Python virtual environment
python3.9 -m venv gptopt

# Activate the virtual environment
source gptopt/bin/activate

# Install the necessary packages
python3.9 -m pip install -e .

echo "Environment setup complete."