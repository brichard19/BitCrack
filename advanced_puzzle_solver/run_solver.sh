#!/bin/bash

# Скрипт запуска продвинутого Bitcoin Puzzle Solver

echo "🚀 Запуск Bitcoin Puzzle Solver v2.0"
echo "===================================="

# Переходим в директорию проекта
cd /home/roman/src/BitCrack/advanced_puzzle_solver

# Проверяем виртуальную среду
if [ ! -d "puzzle_env" ]; then
    echo "⚠️ Создание виртуальной среды..."
    python3 -m venv puzzle_env
fi

# Активируем виртуальную среду
echo "🔧 Активация виртуальной среды..."
source puzzle_env/bin/activate

# Проверяем зависимости
echo "📦 Проверка зависимостей..."
pip list | grep -q "ecdsa" || {
    echo "⚡ Установка зависимостей..."
    pip install ecdsa mnemonic bitcoin eth-account base58 pycryptodome requests psutil rich colorama click tqdm numpy
}

# Запускаем основную программу
echo ""
echo "🎯 Выберите режим:"
echo "1. 🧮 Полнофункциональный решатель"
echo "2. ⚡ Высокопроизводительный решатель"
echo ""

read -p "Введите номер (1-2): " choice

case $choice in
    1)
        echo "🧮 Запуск полнофункционального решателя..."
        python3 puzzle_solver.py
        ;;
    2)
        echo "⚡ Запуск высокопроизводительного решателя..."
        python3 high_performance_solver.py
        ;;
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac
