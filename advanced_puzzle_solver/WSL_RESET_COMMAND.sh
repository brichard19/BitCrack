#!/bin/bash
echo "🔧 WSL COMPLETE RESET - BITCOIN READY"
echo "===================================="

# Полный сброс всех процессов
sudo pkill -f python 2>/dev/null || true
sudo pkill -f pip 2>/dev/null || true

# Очистка всех временных файлов
rm -rf __pycache__ .pytest_cache *.pyc *.pyo 2>/dev/null || true
rm -rf logs/* reports/* results/* 2>/dev/null || true

# Удаляем проблемное окружение если есть
rm -rf puzzle_env 2>/dev/null || true

# Создаем чистое окружение
python3 -m venv puzzle_env
source puzzle_env/bin/activate

# Обновляем pip
pip install --upgrade pip

# Устанавливаем ТОЛЬКО проверенные библиотеки
echo "📦 Installing PROVEN libraries..."
pip install ecdsa==0.18.0
pip install base58==2.1.1  
pip install pycryptodome==3.19.0
pip install requests==2.31.0

# Тест установки
echo "✅ Testing installation..."
python3 -c "
import ecdsa
import base58
from Crypto.Hash import RIPEMD160
import requests
print(\"✅ All proven libraries work!\")
"

# Быстрый тест производительности
echo "🧪 Performance test..."
python3 -c "
import time
import ecdsa
import hashlib

start = time.time()
for i in range(1000):
    sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
elapsed = time.time() - start
print(f\"⚡ Speed: {1000/elapsed:.0f} keys/second\")
"

echo ""
echo "✅ WSL RESET COMPLETE!"
echo "🚀 Ready to run Bitcoin Puzzle Solver!"
echo ""
echo "COMMANDS FOR WSL:"
echo "1. cd /home/roman/src/BitCrack/advanced_puzzle_solver" 
echo "2. source puzzle_env/bin/activate"
echo "3. python3 working_btc_solver.py    # PROVEN version"
echo "4. python3 mega_fast.py             # FAST version"
