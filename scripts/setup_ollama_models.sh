#!/bin/bash
# Setup script for Ollama models on Mac Studio with 128GB RAM
# These models are optimized for high-memory systems

set -e

echo "=== Recommended Ollama Models for 128GB Mac Studio ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Install it first:"
    echo "  brew install ollama"
    echo "  OR"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3
fi

echo "Pulling recommended models..."
echo ""

# Large reasoning models (excellent for complex tasks)
echo "[1/5] Pulling qwen2.5:72b (~40GB) - Best open-source reasoning model"
ollama pull qwen2.5:72b

echo "[2/5] Pulling deepseek-r1:70b (~40GB) - Strong reasoning with chain-of-thought"
ollama pull deepseek-r1:70b

# General purpose large models
echo "[3/5] Pulling llama3.3:70b (~40GB) - Meta's flagship model"
ollama pull llama3.3:70b

# Coding specialist
echo "[4/5] Pulling qwen2.5-coder:32b (~18GB) - Best for code generation"
ollama pull qwen2.5-coder:32b

# Fast medium model for quick tasks
echo "[5/5] Pulling llama3.2:3b (~2GB) - Fast model for simple tasks"
ollama pull llama3.2:3b

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Installed models:"
ollama list
echo ""
echo "Recommended model choices:"
echo "  - qwen2.5:72b       : Best overall reasoning and instruction following"
echo "  - deepseek-r1:70b   : Strong math/reasoning with visible chain-of-thought"
echo "  - llama3.3:70b      : Great general purpose, good for agents"
echo "  - qwen2.5-coder:32b : Best for code generation/review"
echo "  - llama3.2:3b       : Quick responses for simple tasks"
