#!/usr/bin/env bash
# Setup script for Original Raspberry Pi Zero (ARMv6 32-bit) running Raspberry Pi OS Lite
set -euo pipefail

echo "=== Updating apt package index ==="
sudo apt-get update

echo "=== Installing ARMv6-compiled system packages for Vision & GPIO ==="
sudo apt-get install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-pytest \
    python3-gpiozero \
    python3-pigpio \
    pigpio

echo "=== Enabling pigpiod hardware PWM daemon for jitter-free servo control ==="
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

echo "=== Verifying Python imports ==="
python3 -c "import cv2, numpy, yaml, gpiozero; print('OpenCV version:', cv2.__version__); print('Setup complete!')"

