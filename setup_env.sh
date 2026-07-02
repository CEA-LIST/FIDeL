#!/bin/bash
set -e

echo "==========================================================="
echo "Setting up Python Virtual Environment for FIDeL"
echo "==========================================================="

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Warn about PyTorch
echo "==========================================================="
echo "NOTE: PyTorch is NOT installed automatically by this script."
echo "Please install it manually according to your CUDA version:"
echo "https://pytorch.org/get-started/locally/"
echo "==========================================================="
sleep 2

# Install requirements
pip install -r src/requirements.txt

echo "==========================================================="
echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "==========================================================="
