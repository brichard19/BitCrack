#!/bin/bash
echo "🔥 FINAL WSL TEST - MEGA FAST SOLVER"
echo "=================================="
echo "System: $(uname -a)"
echo "CPU: $(nproc) cores"  
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo ""
source puzzle_env/bin/activate
echo "Python: $(python3 --version)"
echo ""
echo "🚀 Launching MEGA FAST solver..."
echo "Time limit: 2 minutes"
echo "Press Ctrl+C to stop early"
echo ""
timeout 120s python3 mega_fast.py
echo ""
echo "✅ Test completed!"
