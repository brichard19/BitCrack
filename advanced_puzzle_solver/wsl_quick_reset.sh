#!/bin/bash
# 🔥 ПРОСТАЯ команда для WSL - быстрый сброс и тест

echo "🔥 WSL QUICK RESET & TEST"
echo "========================"

# Останавливаем все Python процессы
killall python3 2>/dev/null || true

# Очищаем кеш
rm -rf __pycache__ *.pyc 2>/dev/null || true

# Активируем окружение
source puzzle_env/bin/activate 2>/dev/null || {
    echo "Создаю новое окружение..."
    python3 -m venv puzzle_env
    source puzzle_env/bin/activate
}

# Быстрая установка только нужных библиотек
pip install -q coincurve base58 requests psutil

echo "✅ Готово! Запускаю быстрый тест..."

# Простой тест генерации Bitcoin ключей
python3 -c "
import time
import hashlib
import os
from coincurve import PrivateKey

print(\"🚀 Bitcoin Key Generation Speed Test\")
print(\"=\" * 40)

cpu_count = os.cpu_count()
print(f\"CPU cores: {cpu_count}\")

# Тестируем скорость генерации
start_time = time.time()
for i in range(10000):
    private_key = PrivateKey()
    public_key = private_key.public_key
    
    # Быстрое создание адреса
    pubkey_bytes = public_key.format(compressed=True)
    sha256_hash = hashlib.sha256(pubkey_bytes).digest()
    
elapsed = time.time() - start_time
rate = 10000 / elapsed

print(f\"⚡ Speed: {rate:.0f} keys/second\")
print(f\"🔥 With {cpu_count} cores: ~{rate * cpu_count:.0f} keys/sec theoretical\")
print(\"✅ Bitcoin libraries working perfectly!\")
"

echo ""
echo "🎯 ВСЁ ГОТОВО ДЛЯ WSL!"
echo "Теперь можете запускать:"
echo "  python3 mega_fast.py     # Простой быстрый"  
echo "  python3 real_btc_solver.py   # Настоящий Bitcoin"
echo "  python3 ultimate_btc_solver.py  # Максимум функций"
