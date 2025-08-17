#!/bin/bash

echo "🔥 ULTIMATE PERFORMANCE LAUNCHER 🔥"
echo "Запуск максимально производительной версии..."

# Переход в директорию
cd /home/roman/src/BitCrack/advanced_puzzle_solver

# Активация виртуального окружения
source puzzle_env/bin/activate

# Установка базовых зависимостей
echo "📦 Проверка зависимостей..."
pip install --quiet --upgrade pip

# Системная оптимизация
echo "⚙️ Оптимизация системы..."
ulimit -n 65536
ulimit -u 32768

# Проверка ресурсов
echo "💻 Системные ресурсы:"
python3 -c "
import multiprocessing as mp
import os
print(f'🔥 CPU ядер: {mp.cpu_count()}')
print(f'🧵 Максимум потоков: {mp.cpu_count() * 3}')
print(f'💾 PID: {os.getpid()}')
"

echo
echo "🚀 ЗАПУСК ULTIMATE PERFORMANCE SOLVER..."
echo "⚠️  Для остановки используйте Ctrl+C"
echo

# Запуск с высоким приоритетом
nice -n -10 python3 ultimate_performance.py

echo
echo "🏁 Solver завершен!"
echo "📊 Проверьте файлы отчетов"
