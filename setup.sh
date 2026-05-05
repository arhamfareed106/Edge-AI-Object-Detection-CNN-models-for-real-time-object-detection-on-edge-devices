#!/bin/bash
# Setup Script for Edge AI Object Detection Project
# Run this on Raspberry Pi 4 or any Debian-based Linux system

set -e  # Exit on error

echo "=================================================="
echo "Edge AI Object Detection - Setup Script"
echo "Raspberry Pi 4 - Lightweight CNN Evaluation"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo: sudo bash setup.sh"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Update system packages
log_info "[1/7] Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Step 2: Install system dependencies
log_info "[2/7] Installing system dependencies..."
apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libilmbase-dev \
    libopenexr-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libwebp-dev \
    pkg-config \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    tzdata

# Remove unnecessary packages to save space
apt-get autoremove -y

# Step 3: Create project directory
log_info "[3/7] Setting up project directory..."
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create directories
mkdir -p models
mkdir -p datasets/cifar10
mkdir -p datasets/pascal_voc
mkdir -p output/csv
mkdir -p output/graphs
mkdir -p output/images
mkdir -p output/logs
mkdir -p src

log_info "Project directory: $PROJECT_DIR"

# Step 4: Create Python virtual environment
log_info "[4/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Step 5: Install Python packages
log_info "[5/7] Installing Python packages..."

# Install TensorFlow (optimized for Raspberry Pi)
log_info "Installing TensorFlow Lite runtime..."
pip install tensorflow==2.13.0 || {
    log_warn "Full TensorFlow installation failed, trying lite runtime..."
    pip install tensorflow-lite-runtime==2.13.0
}

# Install core dependencies
pip install \
    tensorflow-model-optimization==0.7.4 \
    opencv-python==4.8.0 \
    opencv-contrib-python==4.8.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    psutil==5.9.5 \
    pyyaml==6.0.1 \
    tqdm==4.65.0 \
    scikit-learn==1.3.0 \
    Pillow==10.0.0 \
    scipy==1.11.2

# Optional: Install picamera2 if available
log_info "Attempting to install picamera2..."
pip install picamera2 || log_warn "picamera2 installation skipped (not available for this platform)"

# Optional: Install Edge TPU support
log_info "Attempting to install pycoral..."
pip install pycoral || log_warn "pycoral installation skipped (Edge TPU not detected)"

# Step 6: Enable camera interface (Raspberry Pi specific)
log_info "[6/7] Configuring camera interface..."

# Check if Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    log_info "Detected: $PI_MODEL"
    
    # Enable camera in raspi-config (non-interactive)
    if command -v raspi-config &> /dev/null; then
        log_info "Enabling camera interface..."
        raspi-config nonint do_camera 0
        log_info "Camera interface enabled"
    else
        log_warn "raspi-config not found, skipping camera configuration"
    fi
    
    # Set GPU memory split
    if grep -q "gpu_mem" /boot/config.txt; then
        sudo sed -i 's/gpu_mem=.*/gpu_mem=128/' /boot/config.txt
    else
        echo "gpu_mem=128" >> /boot/config.txt
    fi
    log_info "GPU memory set to 128MB"
else
    log_warn "Not running on Raspberry Pi, skipping Pi-specific configuration"
fi

# Step 7: Set permissions and finalize
log_info "[7/7] Setting permissions..."

# Make scripts executable
chmod +x run_experiments.sh 2>/dev/null || true

# Set ownership
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR" 2>/dev/null || true

# Create .env file for activation
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated"
echo "Run: python src/main.py --help"
EOF

chmod +x activate_env.sh

# Display summary
echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run experiments:"
echo "     python src/main.py --mode full_experiment"
echo ""
echo "  3. Or run automated script:"
echo "     bash run_experiments.sh"
echo ""
echo "For more information, see README.md"
echo "=================================================="
echo ""

# Test installation
log_info "Testing installation..."
python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import tensorflow as tf
    print(f'✓ TensorFlow: {tf.__version__}')
except ImportError:
    print('✗ TensorFlow not installed')
try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except ImportError:
    print('✗ OpenCV not installed')
try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError:
    print('✗ NumPy not installed')
try:
    import pandas as pd
    print(f'✓ Pandas: {pd.__version__}')
except ImportError:
    print('✗ Pandas not installed')
"

echo ""
log_info "Setup complete! You can now run experiments."
echo "=================================================="
