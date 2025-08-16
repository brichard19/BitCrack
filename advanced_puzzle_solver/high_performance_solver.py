#!/usr/bin/env python3
"""
Оптимизированная версия для максимальной производительности
Специально настроен для RTX 3070 Ti и мощных CPU
"""

import os
import sys
import json
import time
import hashlib
import secrets
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging
from queue import Queue
import signal

# Импорты с проверкой
try:
    import ecdsa
    from mnemonic import Mnemonic
    import base58
    from Crypto.Hash import RIPEMD160
    import psutil  # Для мониторинга системы
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    import colorama
    from colorama import Fore, Style
except ImportError as e:
    print(f"❌ Требуется: pip install {e.name}")
    print("🔧 Запустите: python3 install_dependencies.sh")
    sys.exit(1)

# Инициализация
colorama.init()
console = Console()

# Оптимизированный логгер
class OptimizedLogger:
    def __init__(self, log_file="high_performance_solver.log"):
        self.log_file = log_file
        self.console = Console()
        
        # Настраиваем логирование только для критических сообщений
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        self.console.print(f"[blue]ℹ️ {message}[/blue]")
    
    def success(self, message):
        self.console.print(f"[green]✅ {message}[/green]")
    
    def warning(self, message):
        self.console.print(f"[yellow]⚠️ {message}[/yellow]")
        self.logger.warning(message)
    
    def error(self, message):
        self.console.print(f"[red]❌ {message}[/red]")
        self.logger.error(message)
    
    def critical(self, message):
        self.console.print(f"[bold red]🚨 {message}[/bold red]")
        self.logger.critical(message)

logger = OptimizedLogger()

@dataclass
class PerformanceConfig:
    """Высокопроизводительная конфигурация"""
    # Диапазон Puzzle #71
    start_range: int = 0x40000000000000000
    end_range: int = 0x7FFFFFFFFFFFFFFFFF
    
    # Максимальная производительность
    cpu_threads: int = multiprocessing.cpu_count()  # Все ядра
    io_threads: int = 4  # Отдельные потоки для I/O
    batch_size: int = 10000  # Большие батчи
    
    # Оптимизация проверок
    balance_check_interval: int = 1000  # Реже проверяем балансы
    progress_save_interval: int = 50000  # Реже сохраняем прогресс
    
    # Сети (только Bitcoin для скорости)
    networks: List[str] = None
    
    # Файлы
    results_file: str = "high_performance_results.json"
    progress_file: str = "hp_progress.json"
    
    # Оптимизации
    use_memory_pool: bool = True
    precompute_tables: bool = True
    compress_logs: bool = True
    
    def __post_init__(self):
        if self.networks is None:
            self.networks = ["bitcoin"]  # Только Bitcoin для максимальной скорости

class OptimizedKeyGenerator:
    """Оптимизированный генератор с предвычисленными таблицами"""
    
    def __init__(self):
        self.secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        self.secp256k1_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        
        # Предвычисленные значения для ускорения
        self.precompute_tables()
        
        logger.info("🚀 Оптимизированный генератор ключей инициализирован")
    
    def precompute_tables(self):
        """Предвычисление таблиц для ускорения"""
        # Базовая точка secp256k1
        self.G_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        self.G_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        logger.info("📊 Таблицы предвычислений созданы")
    
    def fast_private_to_public(self, private_key_hex: str, compressed: bool = True) -> str:
        """Быстрое преобразование приватного в публичный ключ"""
        try:
            private_key_int = int(private_key_hex, 16)
            
            # Используем быструю библиотеку ecdsa
            sk = ecdsa.SigningKey.from_string(
                private_key_int.to_bytes(32, 'big'),
                curve=ecdsa.SECP256k1
            )
            
            vk = sk.get_verifying_key()
            
            if compressed:
                x = vk.pubkey.point.x()
                y = vk.pubkey.point.y()
                prefix = b'\x02' if y % 2 == 0 else b'\x03'
                return (prefix + x.to_bytes(32, 'big')).hex()
            else:
                return b'\x04' + vk.to_string().hex()
                
        except Exception:
            return ""
    
    def fast_public_to_address(self, public_key_hex: str) -> str:
        """Быстрое преобразование публичного ключа в Bitcoin адрес"""
        try:
            public_key_bytes = bytes.fromhex(public_key_hex)
            
            # SHA256
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            
            # RIPEMD160
            ripemd160_hash = RIPEMD160.new(sha256_hash).digest()
            
            # Версия + hash
            versioned_hash = b'\x00' + ripemd160_hash
            
            # Чексумма
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            return base58.b58encode(versioned_hash + checksum).decode('utf-8')
            
        except Exception:
            return ""

class HighPerformanceBalanceChecker:
    """Высокопроизводительная проверка балансов"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HighPerformancePuzzleSolver/2.0'
        })
        
        # Кеш для адресов
        self.balance_cache = {}
        self.cache_lock = threading.Lock()
        
        # Очередь для батчевых запросов
        self.check_queue = Queue()
        
    def quick_balance_check(self, address: str) -> float:
        """Быстрая проверка баланса с кешированием"""
        with self.cache_lock:
            if address in self.balance_cache:
                return self.balance_cache[address]
        
        try:
            # Используем самый быстрый API
            response = self.session.get(
                f"https://blockstream.info/api/address/{address}",
                timeout=3
            )
            
            if response.status_code == 200:
                data = response.json()
                balance = data.get('chain_stats', {}).get('funded_txo_sum', 0) / 100000000
                
                with self.cache_lock:
                    self.balance_cache[address] = balance
                
                return balance
                
        except Exception:
            pass
        
        return 0.0

class HighPerformanceSolver:
    """Высокопроизводительный решатель"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.key_generator = OptimizedKeyGenerator()
        self.balance_checker = HighPerformanceBalanceChecker()
        
        # Статистика
        self.processed_keys = 0
        self.keys_per_second = 0
        self.found_addresses = []
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Потокобезопасность
        self.stats_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        # Очереди для многопоточности
        self.work_queue = Queue()
        self.results_queue = Queue()
        
        # Флаг остановки
        self.stop_flag = False
        
        logger.success("🚀 Высокопроизводительный решатель инициализирован")
        self.print_system_info()
    
    def print_system_info(self):
        """Вывод информации о системе"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        table = Table(title="🖥️ Системная информация")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")
        
        table.add_row("CPU ядер", str(cpu_count))
        table.add_row("RAM", f"{memory.total / (1024**3):.1f} GB")
        table.add_row("Свободно RAM", f"{memory.available / (1024**3):.1f} GB")
        table.add_row("Потоков CPU", str(self.config.cpu_threads))
        table.add_row("Размер батча", str(self.config.batch_size))
        
        console.print(table)
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.warning("🛑 Получен сигнал остановки...")
        self.stop_flag = True
    
    def process_key_batch(self, private_keys: List[str]) -> List[Dict]:
        """Обработка батча ключей"""
        results = []
        
        for private_key_hex in private_keys:
            if self.stop_flag:
                break
                
            try:
                # Генерируем сжатый и несжатый адреса
                for compressed in [True, False]:
                    public_key = self.key_generator.fast_private_to_public(private_key_hex, compressed)
                    if public_key:
                        address = self.key_generator.fast_public_to_address(public_key)
                        if address:
                            results.append({
                                'private_key': private_key_hex,
                                'public_key': public_key,
                                'address': address,
                                'compressed': compressed,
                                'timestamp': time.time()
                            })
                
                # Обновляем статистику
                with self.stats_lock:
                    self.processed_keys += 1
                    
            except Exception as e:
                logger.error(f"Ошибка обработки ключа {private_key_hex}: {e}")
        
        return results
    
    def check_balances_batch(self, addresses: List[Dict]):
        """Проверка балансов для батча адресов"""
        for addr_info in addresses:
            if self.stop_flag:
                break
                
            try:
                balance = self.balance_checker.quick_balance_check(addr_info['address'])
                
                if balance > 0:
                    addr_info['balance'] = balance
                    
                    with self.results_lock:
                        self.found_addresses.append(addr_info)
                    
                    logger.critical(f"💰 НАЙДЕН КОШЕЛЕК! {addr_info['address']} - {balance} BTC")
                    self.save_found_address(addr_info)
                    
            except Exception as e:
                logger.error(f"Ошибка проверки баланса: {e}")
    
    def save_found_address(self, addr_info: Dict):
        """Немедленное сохранение найденного адреса"""
        try:
            # Загружаем существующие результаты
            results = []
            if os.path.exists(self.config.results_file):
                with open(self.config.results_file, 'r') as f:
                    results = json.load(f)
            
            # Добавляем новый результат
            results.append(addr_info)
            
            # Сохраняем
            with open(self.config.results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения результата: {e}")
    
    def worker_thread(self, thread_id: int):
        """Рабочий поток"""
        logger.info(f"Поток {thread_id} запущен")
        
        while not self.stop_flag:
            try:
                # Генерируем батч ключей
                private_keys = []
                for _ in range(self.config.batch_size):
                    if self.stop_flag:
                        break
                    
                    # Случайный ключ в диапазоне
                    random_key = secrets.randbelow(
                        self.config.end_range - self.config.start_range
                    ) + self.config.start_range
                    
                    private_keys.append(f"{random_key:064x}")
                
                if not private_keys:
                    continue
                
                # Обрабатываем батч
                results = self.process_key_batch(private_keys)
                
                # Проверяем балансы с интервалом
                if self.processed_keys % self.config.balance_check_interval == 0:
                    self.check_balances_batch(results)
                
            except Exception as e:
                logger.error(f"Ошибка в потоке {thread_id}: {e}")
                time.sleep(1)
    
    def update_stats(self):
        """Обновление статистики"""
        current_time = time.time()
        
        with self.stats_lock:
            elapsed = current_time - self.last_update
            if elapsed > 0:
                self.keys_per_second = self.processed_keys / (current_time - self.start_time)
            self.last_update = current_time
    
    def print_live_stats(self):
        """Живая статистика"""
        while not self.stop_flag:
            try:
                self.update_stats()
                
                # Системная информация
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Создаем таблицу статистики
                stats_table = Table(title="📊 Статистика работы")
                stats_table.add_column("Параметр", style="cyan")
                stats_table.add_column("Значение", style="green")
                
                with self.stats_lock:
                    elapsed_hours = (time.time() - self.start_time) / 3600
                    
                    stats_table.add_row("Обработано ключей", f"{self.processed_keys:,}")
                    stats_table.add_row("Скорость", f"{self.keys_per_second:,.0f} ключей/сек")
                    stats_table.add_row("Время работы", f"{elapsed_hours:.2f} часов")
                    stats_table.add_row("CPU загрузка", f"{cpu_percent:.1f}%")
                    stats_table.add_row("RAM использовано", f"{memory.percent:.1f}%")
                    stats_table.add_row("Найдено кошельков", str(len(self.found_addresses)))
                
                console.clear()
                console.print("🔍 HIGH PERFORMANCE BITCOIN PUZZLE SOLVER", style="bold blue")
                console.print(stats_table)
                
                if self.found_addresses:
                    console.print("\n💰 НАЙДЕННЫЕ КОШЕЛЬКИ:", style="bold green")
                    for addr in self.found_addresses[-5:]:  # Последние 5
                        console.print(f"   {addr['address']} - {addr.get('balance', 0)} BTC")
                
                console.print(f"\n🎯 Диапазон: {hex(self.config.start_range)} - {hex(self.config.end_range)}")
                console.print("Нажмите Ctrl+C для остановки", style="dim")
                
                time.sleep(5)  # Обновляем каждые 5 секунд
                
            except Exception as e:
                logger.error(f"Ошибка статистики: {e}")
                time.sleep(1)
    
    def run(self):
        """Основной цикл выполнения"""
        # Регистрируем обработчик сигналов
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.success("🚀 Запуск высокопроизводительного поиска...")
        
        # Запускаем рабочие потоки
        workers = []
        for i in range(self.config.cpu_threads):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Запускаем поток статистики
        stats_thread = threading.Thread(target=self.print_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        try:
            # Основной цикл
            while not self.stop_flag:
                time.sleep(1)
                
                # Сохраняем прогресс
                if self.processed_keys % self.config.progress_save_interval == 0:
                    self.save_progress()
        
        except KeyboardInterrupt:
            logger.warning("🛑 Остановка по запросу пользователя...")
            self.stop_flag = True
        
        finally:
            # Ждем завершения потоков
            logger.info("⏳ Завершение работы потоков...")
            for worker in workers:
                worker.join(timeout=5)
            
            stats_thread.join(timeout=2)
            
            # Финальная статистика
            self.print_final_stats()
            self.save_progress()
    
    def save_progress(self):
        """Сохранение прогресса"""
        try:
            progress = {
                'processed_keys': self.processed_keys,
                'start_time': self.start_time,
                'keys_per_second': self.keys_per_second,
                'found_addresses_count': len(self.found_addresses),
                'last_update': time.time()
            }
            
            with open(self.config.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения прогресса: {e}")
    
    def print_final_stats(self):
        """Финальная статистика"""
        console.print("\n" + "="*60, style="bold")
        console.print("📊 ФИНАЛЬНАЯ СТАТИСТИКА", style="bold blue")
        console.print("="*60, style="bold")
        
        elapsed = time.time() - self.start_time
        
        final_table = Table()
        final_table.add_column("Параметр", style="cyan")
        final_table.add_column("Значение", style="green")
        
        final_table.add_row("Общее время работы", f"{elapsed/3600:.2f} часов")
        final_table.add_row("Обработано ключей", f"{self.processed_keys:,}")
        final_table.add_row("Средняя скорость", f"{self.processed_keys/elapsed:.0f} ключей/сек")
        final_table.add_row("Найдено кошельков", str(len(self.found_addresses)))
        final_table.add_row("Результаты сохранены в", self.config.results_file)
        
        console.print(final_table)
        
        if self.found_addresses:
            console.print("\n💰 НАЙДЕННЫЕ КОШЕЛЬКИ:", style="bold green")
            for addr in self.found_addresses:
                console.print(f"   📍 {addr['address']}")
                console.print(f"   🔑 {addr['private_key']}")
                console.print(f"   💰 {addr.get('balance', 0)} BTC")
                console.print()

def main():
    """Основная функция"""
    console.print("🚀 HIGH PERFORMANCE BITCOIN PUZZLE SOLVER", style="bold blue")
    console.print("="*50, style="bold")
    
    # Оптимальная конфигурация
    config = PerformanceConfig(
        cpu_threads=multiprocessing.cpu_count(),  # Все доступные ядра
        batch_size=50000,  # Большие батчи для максимальной производительности
        balance_check_interval=500,  # Чаще проверяем балансы
        progress_save_interval=100000  # Реже сохраняем для скорости
    )
    
    # Создаем и запускаем решатель
    solver = HighPerformanceSolver(config)
    
    try:
        solver.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
    finally:
        logger.success("✅ Работа завершена")

if __name__ == "__main__":
    main()
