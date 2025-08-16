#!/bin/bash

# Скрипт запуска продвинутого Bitcoin Puzzle Solver

echo "🚀 Запуск Bitcoin Puzzle Solver v2.1"
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
    pip install -r requirements.txt
}

# Запускаем основную программу
echo ""
echo "🎯 Выберите режим:"
echo "1. 🧮 Полнофункциональный решатель"
echo "2. ⚡ Высокопроизводительный решатель"  
echo "3. 🔧 Интегрированный решатель v2.1 (РЕКОМЕНДУЕТСЯ)"
echo "4. 🧪 Тест компонентов"
echo ""

read -p "Введите номер (1-4): " choice

case $choice in
    1)
        echo "🧮 Запуск полнофункционального решателя..."
        python3 puzzle_solver.py
        ;;
    2)
        echo "⚡ Запуск высокопроизводительного решателя..."
        python3 high_performance_solver.py
        ;;
    3)
        echo "🔧 Запуск интегрированного решателя v2.1..."
        python3 integrated_solver.py
        ;;
    4)
        echo "🧪 Тестирование компонентов..."
        echo "🔧 Тест менеджера конфигурации:"
        python3 config_manager.py
        echo ""
        echo "🔍 Тест валидатора адресов:"
        python3 address_validator.py  
        echo ""
        echo "💰 Тест проверщика балансов:"
        python3 balance_checker.py
        ;;
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac
