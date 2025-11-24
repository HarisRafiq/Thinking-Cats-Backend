#!/bin/bash

# Startup script for Streaming Agent Server

set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "🚀 Starting Thinking Cats Server..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your configuration."
        echo "❌ Exiting. Please configure .env and run again."
        exit 1
    else
        echo "❌ No .env.example file found!"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Load environment variables
echo "🔧 Loading environment variables..."
set -a
source .env
set +a

echo "✅ Setup complete!"
echo ""
echo "🌐 Starting server on ${HOST:-0.0.0.0}:${PORT:-8001}..."
echo ""

# Start the server
python api/main.py
