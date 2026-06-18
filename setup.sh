#!/usr/bin/env bash
set -e

# =============================================================================
# Construction Detector — setup script
# Installs SAM 3 via HuggingFace Transformers + dependencies.
#
# SAM 3 (Sam3Model / Sam3Processor) is available in transformers >= 5.12.0.
# An NVIDIA GPU is strongly recommended; the model runs at 1008px and is heavy.
# =============================================================================

echo "=== Construction Detector setup ==="

PYTHON=${PYTHON:-python3}

echo "[1/5] Creating virtual environment (venv)..."
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

echo "[2/5] Installing PyTorch..."
# Auto-detect platform. Override with TORCH_INDEX_URL=... if needed.
OS="$(uname -s)"
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    echo "  Using TORCH_INDEX_URL=$TORCH_INDEX_URL"
    pip install torch torchvision --index-url "$TORCH_INDEX_URL"
elif [ "$OS" = "Darwin" ]; then
    # macOS: no CUDA. Default PyPI wheels include MPS (Apple Silicon) / CPU.
    echo "  macOS detected -> installing default torch (MPS/CPU)."
    pip install torch torchvision
elif command -v nvidia-smi >/dev/null 2>&1; then
    # Linux/Windows with an NVIDIA GPU -> CUDA 12.1 wheels.
    echo "  NVIDIA GPU detected -> installing CUDA 12.1 torch."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    # Linux/Windows without GPU -> CPU wheels.
    echo "  No GPU detected -> installing CPU-only torch."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[3/5] Installing Transformers (with SAM 3 support)..."
# If transformers>=5.12.0 is not yet on PyPI, install from source:
pip install "transformers>=5.12.0" || \
    pip install git+https://github.com/huggingface/transformers.git

echo "[4/5] Installing remaining Python dependencies..."
pip install -r requirements.txt

echo "[5/5] HuggingFace authentication..."
echo "  SAM 3 weights live in the gated repo 'facebook/sam3'."
echo "  1) Request access at: https://huggingface.co/facebook/sam3"
echo "  2) Run: hf auth login    (or: huggingface-cli login)"
echo ""
echo "=== Done! ==="
echo ""
echo "Activate the environment:  source venv/bin/activate"
echo ""
echo "Quick start:"
echo "  python main.py 'https://www.youtube.com/watch?v=XXXX' -o out.mp4"
echo "  python main.py /path/to/local_video.mp4 -o out.mp4"
echo "  python main.py --demo -o demo_out.mp4"
