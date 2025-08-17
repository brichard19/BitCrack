#!/bin/bash

echo "🚀 Ultra Fast Bitcoin Puzzle Solver - WSL Test"
echo "=============================================="
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "OS: $(uname -a)"
echo ""

cd /home/roman/src/BitCrack/advanced_puzzle_solver

# Активируем окружение
source puzzle_env/bin/activate

echo "Python: $(python3 --version)"
echo ""
echo "🔥 ЗАПУСК ULTRA FAST SOLVER"
echo "Для остановки: Ctrl+C"
echo "Время выполнения: 5 минут"
echo ""

# Запускаем на 5 минут
timeout 300s python3 ultra_fast.py

echo ""
echo "✅ Тест завершен!"
echo "Проверьте файлы FOUND_KEY_* если найдены ключи"
ls -la FOUND_KEY_* 2>/dev/null || echo "Ключи не найдены в этой сессии"
