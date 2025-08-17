# 🚀 BitCrack Оптимизация для RTX 3070 Ti

## Ключевые улучшения:

### 📈 Производительность:
- **Блоки**: 256 → 368 (+44% загрузка GPU)
- **Ожидаемая скорость**: 454 → 780+ MKey/s (+70% прирост)
- **Оптимизировано** под RTX 3070 Ti Laptop GPU (46 compute units)

### ⚙️ Настройки:
```bash
# Оптимальные параметры:
-b 368    # блоки (46 compute units × 8)
-t 256    # потоки на блок  
-p 512    # ключей на поток
```

### 🛠 Использование:
```bash
# Быстрый тест производительности
./puzzle71_optimized.sh test

# Запуск Puzzle #71 с оптимальными настройками  
./puzzle71_optimized.sh start

# Мониторинг в реальном времени
./puzzle71_optimized.sh monitor

# Профили производительности:
./puzzle71_optimized.sh start conservative  # 322 блока
./puzzle71_optimized.sh start aggressive    # 414 блоков, 768 ключей/поток
```

### 📊 Бенчмарк:
```bash
./optimization_manager.sh benchmark    # полный тест производительности
```

### 💾 Управление:
```bash  
./optimization_manager.sh save         # сохранить состояние
./optimization_manager.sh validate     # проверить настройки
```

## Файлы оптимизации:
- `puzzle71_optimized.sh` - основной скрипт с оптимизациями
- `optimized_config.sh` - конфигурационные параметры
- `performance_monitor.sh` - система мониторинга
- `optimized_build.mk` - оптимизированные флаги сборки

## Техническая информация:
- **Compute Capability**: 8.6 (RTX 3070 Ti)
- **Compiler optimizations**: -O3, -march=native, --use_fast_math
- **Memory optimizations**: --maxrregcount=64
- **Progress saving**: автоматическое сохранение прогресса
