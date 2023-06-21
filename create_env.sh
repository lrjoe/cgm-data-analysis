#!/bin/bash

echo "Installing Virtual Environment"
pip install virtualenv

echo "Creating glucose_venv Virtual Environment"
virtualenv glucose_venv

echo "Activating Virtual Environment"
source glucose_venv/bin/activate

echo "Installing requirements.txt into Virtual Environment"
pip install -r requirements.txt

echo "Updating apt"
sudo apt update

echo "Installing JDK-17"
sudo apt install openjdk-17-jre-headless -y

echo "Creating `glucose_venv` Kernel"
ipython kernel install --name "glucose-venv" --user

# Run this command in case you want to go back to base environment after activation
# deactivate