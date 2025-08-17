#!/bin/bash

echo "🔥🔥🔥 ULTIMATE BITCOIN PUZZLE SOLVER LAUNCHER 🔥🔥🔥"
echo "⚡ Максимальная производительность активируется..."
echo

# Активировать виртуальное окружение
source /home/roman/src/BitCrack/advanced_puzzle_solver/puzzle_env/bin/activate

# Установить дополнительные зависимости для максимальной производительности
echo "📦 Установка зависимостей для максимальной производительности..."
pip install --upgrade pip
pip install psutil requests ecdsa base58 numba cython asyncio
pip install --upgrade urllib3 requests

# Оптимизация системы
echo "⚙️ Оптимизация системы для максимальной производительности..."

# Увеличить лимиты системы
ulimit -n 65536  # Увеличить лимит файловых дескрипторов
ulimit -u 32768  # Увеличить лимит процессов

# Установить максимальный приоритет процесса
echo "🚀 Установка максимального приоритета..."

# Установить CPU governor на performance (если возможно)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "⚡ Включение максимальной частоты CPU..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [ -w "$cpu" ]; then
            echo performance | sudo tee "$cpu" > /dev/null 2>&1
        fi
    done
fi

# Проверить системные ресурсы
echo "💻 Анализ системных ресурсов:"
python3 -c "
import psutil
print(f'🖥️  CPU: {psutil.cpu_count(logical=True)} ядер')
print(f'🧠 RAM: {psutil.virtual_memory().total // (1024**3)} GB')
print(f'💾 Диск: {psutil.disk_usage(\"/\").free // (1024**3)} GB свободно')
print(f'🔥 CPU частота: {psutil.cpu_freq().max:.0f} MHz')
"

echo
echo "🎯 ЗАПУСК ULTIMATE SOLVER..."
echo "⚠️  ВНИМАНИЕ: Будет использована вся мощность системы!"
echo "⚠️  Для остановки используйте Ctrl+C"
echo

# Запуск с максимальным приоритетом
nice -n -20 python3 /home/roman/src/BitCrack/advanced_puzzle_solver/ultimate_solver.py

echo
echo "🏁 Ultimate Solver завершен!"
echo "📊 Проверьте файлы отчетов для результатов"
