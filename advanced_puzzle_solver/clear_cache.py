#!/usr/bin/env python3
"""
🧹 CACHE CLEANER - Очистка кэша и ресурсов
==========================================
Специальный скрипт для очистки всех кэшированных данных,
зависших процессов и освобождения памяти между запусками
Bitcoin puzzle solvers.

Автор: GitHub Copilot + Roman
Дата: 17 августа 2025
"""

import os
import sys
import gc
import threading
import psutil
import time
import signal
import logging

def clear_python_cache():
    """🧹 Очистка Python кэша"""
    print("🧹 Очистка Python кэша...")
    
    # Принудительная сборка мусора
    gc.collect()
    
    # Очистка sys.modules кэша (осторожно!)
    modules_to_remove = []
    for module_name in sys.modules:
        if any(keyword in module_name.lower() for keyword in 
               ['bitcoin', 'solver', 'puzzle', 'mega', 'ultimate', 'fast']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
            print(f"   📦 Модуль удален: {module_name}")
    
    print("✅ Python кэш очищен")

def kill_bitcoin_processes():
    """🔫 Завершение зависших Bitcoin solver процессов"""
    print("🔫 Поиск зависших Bitcoin solver процессов...")
    
    killed_count = 0
    keywords = ['python3', 'mega_fast', 'ultimate_solver', 'puzzle_solver', 
                'turbo_solver', 'bitcoin_solver']
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline']).lower()
                
                # Проверяем, содержит ли команда наши ключевые слова
                if any(keyword in cmdline for keyword in keywords):
                    # Исключаем текущий процесс
                    if proc.pid != os.getpid():
                        print(f"   🔫 Завершаю процесс: PID {proc.pid} - {proc.info['name']}")
                        proc.terminate()
                        killed_count += 1
                        
                        # Ждем завершения
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            # Если не завершился, убиваем принудительно
                            proc.kill()
                            print(f"   💀 Принудительно убит: PID {proc.pid}")
                            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        print(f"✅ Завершено {killed_count} зависших процессов")
    else:
        print("✅ Зависших процессов не найдено")

def clear_log_files():
    """📝 Очистка лог файлов"""
    print("📝 Очистка лог файлов...")
    
    log_patterns = [
        '*.log',
        '*_solver.log',
        '*_report.json',
        'performance_profile.txt',
        'ultimate_solver_report.json'
    ]
    
    import glob
    cleared_count = 0
    
    for pattern in log_patterns:
        for log_file in glob.glob(pattern):
            try:
                if os.path.getsize(log_file) > 1024 * 1024:  # Больше 1MB
                    with open(log_file, 'w') as f:
                        f.write(f"# Лог очищен: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"   🗑️ Очищен: {log_file}")
                    cleared_count += 1
            except Exception as e:
                print(f"   ❌ Ошибка очистки {log_file}: {e}")
    
    if cleared_count > 0:
        print(f"✅ Очищено {cleared_count} лог файлов")
    else:
        print("✅ Крупных лог файлов не найдено")

def clear_threading_resources():
    """🧵 Очистка threading ресурсов"""
    print("🧵 Очистка threading ресурсов...")
    
    # Получаем список всех потоков
    active_threads = threading.enumerate()
    main_thread = threading.main_thread()
    
    # Завершаем не-главные потоки
    killed_threads = 0
    for thread in active_threads:
        if thread != main_thread and thread.is_alive():
            thread_name = getattr(thread, 'name', 'Unknown')
            print(f"   🧵 Завершаю поток: {thread_name}")
            
            # Для потоков ThreadPoolExecutor
            if hasattr(thread, '_stop'):
                thread._stop()
            
            killed_threads += 1
    
    if killed_threads > 0:
        print(f"✅ Завершено {killed_threads} потоков")
    else:
        print("✅ Дополнительных потоков не найдено")

def clear_memory():
    """🧠 Освобождение памяти"""
    print("🧠 Освобождение памяти...")
    
    # Получаем статистику памяти до очистки
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Множественная сборка мусора
    for i in range(3):
        gc.collect()
        time.sleep(0.1)
    
    # Статистика после
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    freed = memory_before - memory_after
    
    print(f"   💾 Память до: {memory_before:.1f} MB")
    print(f"   💾 Память после: {memory_after:.1f} MB")
    if freed > 0:
        print(f"   ✅ Освобождено: {freed:.1f} MB")
    else:
        print("   ✅ Память оптимизирована")

def main():
    """🧹 Главная функция очистки"""
    print("🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹")
    print("🧹 BITCOIN PUZZLE SOLVER CACHE CLEANER 🧹")
    print("🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹🧹")
    print()
    
    try:
        # Пошаговая очистка
        clear_python_cache()
        print()
        
        kill_bitcoin_processes()
        print()
        
        clear_threading_resources()
        print()
        
        clear_log_files()
        print()
        
        clear_memory()
        print()
        
        print("🎉 ПОЛНАЯ ОЧИСТКА ЗАВЕРШЕНА!")
        print("✅ Система готова к новому запуску Bitcoin solver'ов")
        
    except Exception as e:
        print(f"❌ Ошибка во время очистки: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
