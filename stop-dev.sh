#!/bin/bash
# Stop Annual Report Analyst dev stack

set -euo pipefail

echo "🛑 Stopping Annual Report Analyst dev stack..."

# Terminate backend processes
echo "🔧 Stopping backend service..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python -m app" 2>/dev/null || true
pkill -f "uvicorn.*8000" 2>/dev/null || true

# Force cleanup processes occupying port 8080
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "🔧 Cleaning up processes occupying port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

# Terminate frontend processes
echo "🎨 Stopping frontend service..."
pkill -f "vite" 2>/dev/null || true
pkill -f "pnpm dev" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

if lsof -ti:5173 >/dev/null 2>&1; then
    echo "🔧 Cleaning up processes occupying port 5173..."
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
fi

# Wait for processes to completely terminate
sleep 2

echo "✅ All services stopped"
