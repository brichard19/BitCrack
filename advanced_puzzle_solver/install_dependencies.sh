#!/bin/bash
"""
Установщик зависимостей для продвинутого Bitcoin Puzzle Solver
"""

echo "🔧 Установка зависимостей для Puzzle Solver..."

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+"
    exit 1
fi

# Обновляем pip
python3 -m pip install --upgrade pip

# Основные криптографические библиотеки
echo "📦 Установка криптографических библиотек..."
python3 -m pip install \
    ecdsa \
    mnemonic \
    bip32 \
    bitcoin \
    eth-account \
    base58 \
    pycryptodome \
    requests \
    web3

# Дополнительные полезные библиотеки
echo "📦 Установка дополнительных библиотек..."
python3 -m pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    colorama \
    rich \
    click \
    tqdm

echo "✅ Все зависимости установлены!"
echo ""
echo "🚀 Для запуска используйте:"
echo "   python3 puzzle_solver.py"
echo ""
echo "📁 Результаты будут сохранены в:"
echo "   - found_wallets.json (найденные кошельки)"
echo "   - solver_progress.json (прогресс)"
echo "   - puzzle_solver.log (логи)"
