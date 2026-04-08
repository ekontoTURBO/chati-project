#!/bin/bash
# ============================================================
# WSL2 Setup Script for Chati — Ollama + Gemma 4 E2B
# Run this INSIDE your WSL2 Ubuntu instance
# Usage: bash setup_wsl.sh
# ============================================================

set -e

echo "============================================"
echo " Chati — WSL2 Environment Setup"
echo "============================================"

# --- Step 1: System packages ---
echo "[1/4] Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl zstd

# --- Step 2: Check NVIDIA GPU ---
echo "[2/4] Checking CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo "Make sure your Windows NVIDIA drivers are v535+."
    echo "WSL2 shares the Windows GPU driver."
    echo "Download from: https://www.nvidia.com/drivers"
    exit 1
fi
nvidia-smi
echo "CUDA is available via Windows driver passthrough."

# --- Step 3: Install Ollama ---
echo "[3/4] Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama already installed: $(ollama --version)"
else
    echo "Downloading Ollama..."
    curl -L -o /tmp/ollama.tar.zst \
        'https://github.com/ollama/ollama/releases/download/v0.20.3/ollama-linux-amd64.tar.zst'
    sudo tar --zstd -xf /tmp/ollama.tar.zst -C /usr
    rm /tmp/ollama.tar.zst
    echo "Ollama installed: $(ollama --version)"
fi

# --- Step 4: Pull Gemma 4 E2B model ---
echo "[4/4] Pulling Gemma 4 E2B model (~7 GB download)..."
echo "Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!
sleep 3

ollama pull gemma4:e2b

echo ""
echo "============================================"
echo " Setup Complete!"
echo ""
echo " To start the server:"
echo "   ollama serve"
echo ""
echo " The OpenAI-compatible API will be at:"
echo "   http://localhost:11434/v1/"
echo ""
echo " Then start the agent on Windows:"
echo "   python -m agent.controller"
echo "============================================"

# Stop the background server
kill $OLLAMA_PID 2>/dev/null
