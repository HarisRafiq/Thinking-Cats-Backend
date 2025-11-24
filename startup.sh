#!/bin/bash

# Startup script for Streaming Agent Server

set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "🚀 Starting Thinking Cats Server..."

 
# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.10 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

 
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting server on ${HOST:-0.0.0.0}:${PORT:-8001}..."
echo ""

# Start the server
python api/main.py
