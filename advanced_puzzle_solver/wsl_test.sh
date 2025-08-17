#!/bin/bash
echo "🚀 Быстрый тест в WSL"
echo "==================="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""
source puzzle_env/bin/activate
echo "Python: $(python3 --version)"
echo "Запускаем fast_test.py..."
timeout 60s python3 fast_test.py
