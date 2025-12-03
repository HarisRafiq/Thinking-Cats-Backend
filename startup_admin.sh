#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the admin server
echo "🚀 Starting Admin Panel Server..."
python api/admin_server.py

