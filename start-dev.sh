#!/bin/bash
# Annual Report Analyst One-Click Startup Script

set -euo pipefail

echo "🚀 Starting Annual Report Analyst dev stack..."

# Terminate potentially running old processes
echo "🧹 Cleaning up old processes..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python -m app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "pnpm dev" 2>/dev/null || true

# Force cleanup processes occupying ports
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "🔧 Cleaning up processes occupying port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi
if lsof -ti:5173 >/dev/null 2>&1; then
    echo "🔧 Cleaning up processes occupying port 5173..."
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
fi

# Wait for processes to completely terminate
sleep 2

# Start backend with virtual environment
echo "🔧 Starting backend service..."
cd backend
if [ ! -d ".venv" ]; then
    echo "⚠️  Creating backend virtual environment (.venv)"
    python3 -m venv .venv
fi
source .venv/bin/activate
INSTALL_SENTINEL=".venv/.deps-installed"
if [ ! -f "$INSTALL_SENTINEL" ]; then
    echo "📦 Installing backend dependencies (first run may take a minute)..."
    pip install --upgrade pip >/dev/null
    pip install -e . >/dev/null
    touch "$INSTALL_SENTINEL"
else
    echo "✅ Backend dependencies already installed; skipping reinstall."
fi
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Start frontend with nvm
echo "🎨 Starting frontend service..."
cd frontend
if command -v pnpm >/dev/null 2>&1; then
    PACKAGE_MANAGER="pnpm"
elif command -v npm >/dev/null 2>&1; then
    PACKAGE_MANAGER="npm"
else
    echo "❌ Neither pnpm nor npm found. Please install one of them."
    exit 1
fi

if [ "$PACKAGE_MANAGER" = "pnpm" ]; then
    pnpm install >/dev/null
    pnpm dev --host 0.0.0.0 --port 5173 &
else
    npm install >/dev/null
    npm run dev -- --host 0.0.0.0 --port 5173 &
fi
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 3

echo ""
echo "✅ Services started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 Frontend App:  http://localhost:5173"
echo "🔧 Backend API:   http://localhost:8000"
echo "📚 API Docs:      http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Process IDs: Backend=$BACKEND_PID, Frontend=$FRONTEND_PID"
echo ""
echo "🔧 Environment Info:"
echo "   Backend: Python virtual environment (backend/.venv)"
echo "   Frontend: Node.js $(cd frontend && node --version 2>/dev/null || echo 'N/A') using $PACKAGE_MANAGER"
echo ""
echo "Stop services: ./stop-dev.sh"
echo "View logs: ./logs-dev.sh"
echo ""
echo "🎉 Development environment ready!"
