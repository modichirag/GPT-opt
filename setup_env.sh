#!/bin/bash

# filepath: /Users/rgower/Code/matsign/setup_env.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Create a Python virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the necessary packages
python3 -m pip install -e .

echo "Environment setup complete."