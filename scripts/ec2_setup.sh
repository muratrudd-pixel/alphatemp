#!/bin/bash
# EC2 instance setup script — run this on a fresh Amazon Linux 2023 / Ubuntu instance.
# Installs Python deps, clones repo, and launches the parallel HRRR backfill.
#
# Usage (on EC2):
#   curl -sSL <raw-github-url> | bash
#   OR
#   scp this file to EC2, then: chmod +x ec2_setup.sh && ./ec2_setup.sh

set -euo pipefail

echo "=== AlphaTemp EC2 Setup ==="
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Detect OS and install system packages
if command -v dnf &>/dev/null; then
    # Amazon Linux 2023
    echo "Detected Amazon Linux — installing via dnf..."
    sudo dnf update -y
    sudo dnf install -y python3.11 python3.11-pip python3.11-devel git gcc gcc-c++ \
        openblas-devel cmake eccodes eccodes-devel
    PYTHON=python3.11
elif command -v apt-get &>/dev/null; then
    # Ubuntu
    echo "Detected Ubuntu — installing via apt..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev git gcc g++ \
        libopenblas-dev cmake libeccodes-dev
    PYTHON=python3
else
    echo "Unsupported OS — install Python 3.9+ manually"
    exit 1
fi

echo "Python: $($PYTHON --version)"

# Clone the repo
cd ~
if [ ! -d "alphatemp" ]; then
    echo "Cloning alphatemp..."
    git clone https://github.com/muratrudd-pixel/alphatemp.git
else
    echo "Repo already exists — pulling latest..."
    cd alphatemp && git pull && cd ~
fi

cd ~/alphatemp

# Create venv and install dependencies
echo "Setting up virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify key imports
echo "Verifying imports..."
python -c "import herbie, pygrib, duckdb, numpy, loguru; print('All imports OK')"

# Create directories
mkdir -p data logs

# Set faster delay for EC2 (co-located with S3)
export HRRR_DELAY=0.1

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the backfill:"
echo "  cd ~/alphatemp"
echo "  source .venv/bin/activate"
echo "  chmod +x scripts/hrrr_parallel_backfill.sh"
echo "  ./scripts/hrrr_parallel_backfill.sh"
echo ""
echo "To monitor:"
echo "  ./scripts/hrrr_parallel_backfill.sh --status"
echo ""
echo "When done, download results:"
echo "  # From your local machine:"
echo "  scp -i your-key.pem ec2-user@<EC2-IP>:~/alphatemp/data/backfill_hrrr_*.duckdb ./data/"
