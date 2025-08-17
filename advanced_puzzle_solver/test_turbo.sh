#!/bin/bash
# Turbo Test Script для WSL
# Быстрый запуск без сложных зависимостей

echo "🚀 Turbo Bitcoin Solver - WSL Test"
echo "=================================="

cd /home/roman/src/BitCrack/advanced_puzzle_solver

# Активируем виртуальное окружение
if [ -d "puzzle_env" ]; then
    echo "📦 Активируем виртуальное окружение..."
    source puzzle_env/bin/activate
else
    echo "❌ Виртуальное окружение не найдено!"
    exit 1
fi

# Показываем системную информацию
echo "💻 Системная информация:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Python version: $(python3 --version)"

# Запускаем turbo solver
echo ""
echo "🏁 Запускаем Turbo Solver..."
echo "Для остановки нажмите Ctrl+C"
echo ""

timeout 300s python3 turbo_solver.py

echo ""
echo "✅ Тест завершен!"
