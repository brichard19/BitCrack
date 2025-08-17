#!/bin/bash
# 🔧 WSL Reset & Modern Bitcoin Libraries Installer
# Сбрасывает все ошибки и устанавливает лучшие библиотеки

echo "🚀 WSL RESET & BITCOIN LIBRARIES INSTALLER"
echo "============================================"

# Переходим в рабочую директорию
cd /home/roman/src/BitCrack/advanced_puzzle_solver

# Сбрасываем все процессы Python
echo "🛑 Stopping all Python processes..."
pkill -f python3 2>/dev/null || true
sleep 2

# Очищаем кеш Python
echo "🗑️  Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Проверяем и активируем виртуальное окружение
if [ ! -d "puzzle_env" ]; then
    echo "📦 Creating new virtual environment..."
    python3 -m venv puzzle_env
fi

echo "🔄 Activating virtual environment..."
source puzzle_env/bin/activate

# Обновляем pip
echo "⬆️  Updating pip..."
pip install --upgrade pip

# Устанавливаем системные зависимости для криптографии
echo "🔧 Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev libsecp256k1-dev

# Устанавливаем современные Bitcoin библиотеки
echo "💎 Installing modern Bitcoin libraries..."

# Основные быстрые библиотеки
pip install secp256k1==0.14.0
pip install coincurve==18.0.0
pip install bitcoin==1.1.42
pip install base58==2.1.1
pip install hdwallet==2.2.1

# Дополнительные полезные библиотеки
pip install ecdsa==0.18.0
pip install mnemonic==0.20
pip install bip32==3.4
pip install bitcoinlib==0.12.0
pip install pycryptodome==3.19.0

# Библиотеки для производительности  
pip install psutil==5.9.6
pip install requests==2.31.0
pip install rich==13.7.0
pip install colorama==0.4.6

# Библиотеки для численных вычислений
pip install numpy==1.25.2
pip install numba==0.58.1  # JIT компилятор для ускорения

# Проверяем установку
echo "✅ Checking installations..."
python3 -c "
import secp256k1; print('✅ secp256k1:', secp256k1.__version__)
import coincurve; print('✅ coincurve:', coincurve.__version__)
import base58; print('✅ base58: OK')
import hdwallet; print('✅ hdwallet: OK')
import bitcoin; print('✅ bitcoin: OK')
import ecdsa; print('✅ ecdsa:', ecdsa.__version__)
import mnemonic; print('✅ mnemonic: OK')
import psutil; print('✅ psutil:', psutil.__version__)
import requests; print('✅ requests:', requests.__version__)
print('🎉 All libraries installed successfully!')
"

# Показываем информацию о системе
echo ""
echo "📊 SYSTEM INFO:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Python: $(python3 --version)"
echo "Pip: $(pip --version)"

# Создаем быстрый тест
echo ""
echo "🧪 Creating quick test..."
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
import time
import secp256k1
import coincurve

print("🧪 Quick Performance Test")
print("=" * 30)

# Test secp256k1
start = time.time()
for i in range(1000):
    private_key = secp256k1.PrivateKey()
    public_key = private_key.pubkey
secp_time = time.time() - start
print(f"secp256k1: {1000/secp_time:.0f} keys/sec")

# Test coincurve
start = time.time()
for i in range(1000):
    private_key = coincurve.PrivateKey()
    public_key = private_key.public_key
coin_time = time.time() - start
print(f"coincurve: {1000/coin_time:.0f} keys/sec")

print("✅ Libraries working perfectly!")
EOF

python3 quick_test.py

echo ""
echo "✅ WSL RESET COMPLETE!"
echo "🚀 Modern Bitcoin libraries installed"
echo "🎯 Ready for maximum performance Bitcoin mining!"
echo ""
echo "Next steps:"
echo "1. source puzzle_env/bin/activate"
echo "2. python3 real_btc_solver.py"
