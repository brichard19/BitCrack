#!/usr/bin/env python3
"""
Интегрированный Bitcoin Puzzle Solver v2.1
Использует все новые компоненты для максимальной надежности
"""

import os
import sys
import json
import time
import hashlib
import secrets
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# Импорт новых компонентов
try:
    from config_manager import get_config_manager, ConfigManager
    from address_validator import get_address_validator, validate_address
    from balance_checker import get_balance_checker, AdvancedBalanceChecker, BalanceResult
except ImportError:
    print("❌ Не найдены модули проекта. Убедитесь что все файлы находятся в одной директории.")
    sys.exit(1)

# Криптографические библиотеки
try:
    import ecdsa
    from mnemonic import Mnemonic
    from Crypto.Hash import RIPEMD160
    import base58
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, track
    from rich.panel import Panel
    from rich.text import Text
except ImportError as e:
    print(f"❌ Требуется установка зависимостей: pip install {e.name}")
    print("🔧 Запустите: ./install_dependencies.sh")
    sys.exit(1)

# Инициализация
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SolverConfig:
    """Конфигурация решателя"""
    threads: int = 8
    batch_size: int = 5000
    balance_check_interval: int = 100
    save_interval: int = 10000
    target_puzzle: str = "71"
    search_mode: str = "mixed"  # sequential, random, mnemonic, mixed
    duration_hours: float = 24.0

class IntegratedKeyGenerator:
    """Интегрированный генератор ключей с валидацией"""
    
    def __init__(self):
        self.mnemonic_generator = Mnemonic("english")
        self.validator = get_address_validator()
        self.processed_keys = 0
        
        console.print("🔐 Генератор ключей инициализирован", style="green")
    
    def generate_private_key_hex(self, key_int: int) -> str:
        """Генерация приватного ключа в hex формате"""
        return f"{key_int:064x}"
    
    def private_to_public_compressed(self, private_key_hex: str) -> str:
        """Генерация сжатого публичного ключа"""
        try:
            private_key_int = int(private_key_hex, 16)
            sk = ecdsa.SigningKey.from_string(
                private_key_int.to_bytes(32, 'big'),
                curve=ecdsa.SECP256k1
            )
            vk = sk.get_verifying_key()
            
            x = vk.pubkey.point.x()
            y = vk.pubkey.point.y()
            prefix = b'\x02' if y % 2 == 0 else b'\x03'
            
            return (prefix + x.to_bytes(32, 'big')).hex()
            
        except Exception as e:
            logger.error(f"Ошибка генерации публичного ключа: {e}")
            return ""
    
    def private_to_public_uncompressed(self, private_key_hex: str) -> str:
        """Генерация несжатого публичного ключа"""
        try:
            private_key_int = int(private_key_hex, 16)
            sk = ecdsa.SigningKey.from_string(
                private_key_int.to_bytes(32, 'big'),
                curve=ecdsa.SECP256k1
            )
            vk = sk.get_verifying_key()
            
            return (b'\x04' + vk.to_string()).hex()
            
        except Exception as e:
            logger.error(f"Ошибка генерации несжатого ключа: {e}")
            return ""
    
    def public_to_bitcoin_address(self, public_key_hex: str) -> str:
        """Генерация Bitcoin адреса из публичного ключа"""
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
            
            # Финальный адрес
            address = base58.b58encode(versioned_hash + checksum).decode('utf-8')
            
            # Валидация сгенерированного адреса
            validation = self.validator.validate_bitcoin_address(address)
            if not validation[0]:
                logger.warning(f"Сгенерирован невалидный адрес: {address}")
                return ""
            
            return address
            
        except Exception as e:
            logger.error(f"Ошибка генерации адреса: {e}")
            return ""
    
    def generate_mnemonic_wallet(self) -> Optional[Dict[str, Any]]:
        """Генерация кошелька из мнемоники"""
        try:
            mnemonic = self.mnemonic_generator.generate(strength=256)
            seed = self.mnemonic_generator.to_seed(mnemonic)
            
            # Используем простую деривацию для скорости
            private_key = hashlib.sha256(seed).digest()
            private_key_hex = private_key.hex()
            
            # Генерируем адрес
            public_key = self.private_to_public_compressed(private_key_hex)
            if not public_key:
                return None
            
            address = self.public_to_bitcoin_address(public_key)
            if not address:
                return None
            
            return {
                'private_key': private_key_hex,
                'public_key': public_key,
                'address': address,
                'mnemonic': mnemonic,
                'type': 'mnemonic'
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации мнемоники: {e}")
            return None

class IntegratedPuzzleSolver:
    """Интегрированный решатель с всеми компонентами"""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        
        # Инициализация компонентов
        self.config_manager = get_config_manager()
        self.key_generator = IntegratedKeyGenerator()
        self.address_validator = get_address_validator()
        self.balance_checker = get_balance_checker()
        
        # Получаем активный puzzle
        self.active_puzzle = self.config_manager.get_active_puzzle()
        if config.target_puzzle == "72":
            puzzle_72 = self.config_manager.get_puzzle_by_name("Puzzle #72")
            if puzzle_72:
                self.active_puzzle = puzzle_72
        
        # Статистика
        self.stats = {
            'processed_keys': 0,
            'generated_addresses': 0,
            'checked_balances': 0,
            'found_wallets': 0,
            'start_time': time.time(),
            'errors': 0
        }
        
        # Потокобезопасность
        self.stats_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.found_wallets = []
        
        # Флаг остановки
        self.stop_flag = False
        
        console.print(Panel.fit(
            f"🎯 Целевой Puzzle: {self.active_puzzle.name}\n"
            f"💰 Награда: {self.active_puzzle.reward_btc} BTC\n"
            f"📍 Адрес: {self.active_puzzle.address}\n"
            f"🔍 Диапазон: {hex(self.active_puzzle.start_range)} - {hex(self.active_puzzle.end_range)}",
            title="🚀 Интегрированный Puzzle Solver v2.1",
            style="bold blue"
        ))
    
    def save_found_wallet(self, wallet_info: Dict[str, Any], balance_result: BalanceResult):
        """Сохранение найденного кошелька"""
        with self.results_lock:
            wallet_data = {
                'address': wallet_info['address'],
                'private_key': wallet_info['private_key'],
                'public_key': wallet_info['public_key'],
                'balance': balance_result.balance,
                'network': balance_result.network,
                'provider': balance_result.provider,
                'found_timestamp': time.time(),
                'type': wallet_info.get('type', 'generated'),
                'mnemonic': wallet_info.get('mnemonic'),
                'is_puzzle_target': self.config_manager.is_target_address(wallet_info['address'])
            }
            
            self.found_wallets.append(wallet_data)
            
            # Сохраняем в файл немедленно
            self._save_results_to_file()
            
            # Если это puzzle адрес - особая обработка
            if wallet_data['is_puzzle_target']:
                self.config_manager.add_found_address(
                    wallet_info['address'], 
                    wallet_info['private_key']
                )
                console.print(Panel.fit(
                    f"🎉 НАЙДЕН PUZZLE АДРЕС!\n"
                    f"📍 Адрес: {wallet_info['address']}\n"
                    f"🔑 Приватный ключ: {wallet_info['private_key']}\n"
                    f"💰 Баланс: {balance_result.balance} {balance_result.network.upper()}\n"
                    f"🏆 Награда: {self.active_puzzle.reward_btc} BTC",
                    title="🎯 PUZZLE РЕШЕН!",
                    style="bold green"
                ))
            else:
                console.print(f"💰 Найден кошелек: {wallet_info['address']} - {balance_result.balance} BTC", style="green")
    
    def _save_results_to_file(self):
        """Сохранение результатов в файл"""
        try:
            results_file = f"integrated_solver_results_{int(time.time())}.json"
            
            # Загружаем существующие результаты если есть
            all_results = []
            if os.path.exists("all_found_wallets.json"):
                try:
                    with open("all_found_wallets.json", 'r') as f:
                        all_results = json.load(f)
                except:
                    pass
            
            # Добавляем новые результаты
            all_results.extend(self.found_wallets)
            
            # Сохраняем
            with open("all_found_wallets.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Сохраняем также отдельный файл для этой сессии
            with open(results_file, 'w') as f:
                json.dump(self.found_wallets, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
    
    def process_key_batch(self, private_keys: List[str]) -> List[Dict[str, Any]]:
        """Обработка батча приватных ключей"""
        results = []
        
        for private_key_hex in private_keys:
            if self.stop_flag:
                break
            
            try:
                # Генерируем сжатый и несжатый адреса
                for compressed in [True, False]:
                    if compressed:
                        public_key = self.key_generator.private_to_public_compressed(private_key_hex)
                    else:
                        public_key = self.key_generator.private_to_public_uncompressed(private_key_hex)
                    
                    if not public_key:
                        continue
                    
                    address = self.key_generator.public_to_bitcoin_address(public_key)
                    if not address:
                        continue
                    
                    results.append({
                        'private_key': private_key_hex,
                        'public_key': public_key,
                        'address': address,
                        'compressed': compressed,
                        'type': 'sequential'
                    })
                    
                    with self.stats_lock:
                        self.stats['generated_addresses'] += 1
                
                with self.stats_lock:
                    self.stats['processed_keys'] += 1
                    
            except Exception as e:
                with self.stats_lock:
                    self.stats['errors'] += 1
                logger.error(f"Ошибка обработки ключа {private_key_hex}: {e}")
        
        return results
    
    def check_balances_for_addresses(self, addresses: List[Dict[str, Any]]):
        """Проверка балансов для списка адресов"""
        # Подготавливаем список для проверки
        check_list = [(addr['address'], 'bitcoin') for addr in addresses]
        
        # Проверяем балансы
        balance_results = self.balance_checker.check_multiple_balances(check_list, max_workers=3)
        
        # Обрабатываем результаты
        for balance_result in balance_results:
            with self.stats_lock:
                self.stats['checked_balances'] += 1
            
            # Находим соответствующий адрес
            matching_addr = None
            for addr_info in addresses:
                if addr_info['address'] == balance_result.address:
                    matching_addr = addr_info
                    break
            
            if matching_addr and (balance_result.balance > 0 or 
                                self.config_manager.is_target_address(balance_result.address)):
                with self.stats_lock:
                    self.stats['found_wallets'] += 1
                
                self.save_found_wallet(matching_addr, balance_result)
    
    def sequential_search_worker(self, start_key: int, end_key: int, thread_id: int):
        """Рабочий поток для последовательного поиска"""
        logger.info(f"Поток {thread_id}: последовательный поиск {hex(start_key)} - {hex(end_key)}")
        
        current_key = start_key
        while current_key <= end_key and not self.stop_flag:
            try:
                # Создаем батч ключей
                batch_keys = []
                batch_end = min(current_key + self.config.batch_size, end_key + 1)
                
                for key_int in range(current_key, batch_end):
                    private_key_hex = self.key_generator.generate_private_key_hex(key_int)
                    batch_keys.append(private_key_hex)
                
                # Обрабатываем батч
                addresses = self.process_key_batch(batch_keys)
                
                # Проверяем балансы с интервалом
                if self.stats['processed_keys'] % self.config.balance_check_interval == 0:
                    self.check_balances_for_addresses(addresses)
                
                current_key = batch_end
                
            except Exception as e:
                logger.error(f"Ошибка в последовательном поиске поток {thread_id}: {e}")
                current_key += self.config.batch_size
    
    def random_search_worker(self, thread_id: int, end_time: float):
        """Рабочий поток для случайного поиска"""
        logger.info(f"Поток {thread_id}: случайный поиск")
        
        while time.time() < end_time and not self.stop_flag:
            try:
                # Генерируем батч случайных ключей
                batch_keys = []
                for _ in range(self.config.batch_size):
                    random_key = secrets.randbelow(
                        self.active_puzzle.end_range - self.active_puzzle.start_range
                    ) + self.active_puzzle.start_range
                    
                    private_key_hex = self.key_generator.generate_private_key_hex(random_key)
                    batch_keys.append(private_key_hex)
                
                # Обрабатываем батч
                addresses = self.process_key_batch(batch_keys)
                
                # Проверяем балансы
                if len(addresses) > 0:
                    self.check_balances_for_addresses(addresses)
                
            except Exception as e:
                logger.error(f"Ошибка в случайном поиске поток {thread_id}: {e}")
                time.sleep(1)
    
    def mnemonic_search_worker(self, thread_id: int, count: int):
        """Рабочий поток для поиска через мнемоники"""
        logger.info(f"Поток {thread_id}: поиск через мнемоники")
        
        for i in range(count):
            if self.stop_flag:
                break
            
            try:
                wallet = self.key_generator.generate_mnemonic_wallet()
                if wallet:
                    # Проверяем баланс
                    balance_result = self.balance_checker.check_balance(wallet['address'], 'bitcoin')
                    
                    with self.stats_lock:
                        self.stats['checked_balances'] += 1
                        self.stats['generated_addresses'] += 1
                        self.stats['processed_keys'] += 1
                    
                    if (balance_result.balance > 0 or 
                        self.config_manager.is_target_address(wallet['address'])):
                        with self.stats_lock:
                            self.stats['found_wallets'] += 1
                        
                        self.save_found_wallet(wallet, balance_result)
                
            except Exception as e:
                logger.error(f"Ошибка в мнемоническом поиске поток {thread_id}: {e}")
    
    def print_stats(self):
        """Вывод статистики"""
        with self.stats_lock:
            elapsed = time.time() - self.stats['start_time']
            
            table = Table(title="📊 Статистика работы")
            table.add_column("Параметр", style="cyan")
            table.add_column("Значение", style="green")
            
            table.add_row("Время работы", f"{elapsed/3600:.2f} часов")
            table.add_row("Обработано ключей", f"{self.stats['processed_keys']:,}")
            table.add_row("Сгенерировано адресов", f"{self.stats['generated_addresses']:,}")
            table.add_row("Проверено балансов", f"{self.stats['checked_balances']:,}")
            table.add_row("Найдено кошельков", str(self.stats['found_wallets']))
            table.add_row("Ошибок", str(self.stats['errors']))
            
            if elapsed > 0:
                table.add_row("Скорость ключей/сек", f"{self.stats['processed_keys']/elapsed:.2f}")
                table.add_row("Скорость адресов/сек", f"{self.stats['generated_addresses']/elapsed:.2f}")
            
            console.clear()
            console.print(table)
            
            # Статистика balance checker
            bc_stats = self.balance_checker.get_stats()
            console.print(f"💰 API статистика: {bc_stats['total_requests']} запросов, "
                        f"успешность {bc_stats['success_rate']:.1f}%")
    
    def run_mixed_search(self):
        """Смешанный поиск - комбинация всех методов"""
        console.print("🔀 Запуск смешанного поиска", style="bold yellow")
        
        end_time = time.time() + self.config.duration_hours * 3600
        
        # Разделяем потоки
        sequential_threads = self.config.threads // 3
        random_threads = self.config.threads // 3  
        mnemonic_threads = self.config.threads - sequential_threads - random_threads
        
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # Последовательные потоки
            range_size = (self.active_puzzle.end_range - self.active_puzzle.start_range) // sequential_threads
            for i in range(sequential_threads):
                start_key = self.active_puzzle.start_range + i * range_size
                end_key = start_key + range_size - 1
                if i == sequential_threads - 1:
                    end_key = self.active_puzzle.end_range
                
                future = executor.submit(self.sequential_search_worker, start_key, end_key, i)
                futures.append(future)
            
            # Случайные потоки
            for i in range(random_threads):
                future = executor.submit(self.random_search_worker, i + sequential_threads, end_time)
                futures.append(future)
            
            # Мнемонические потоки
            mnemonic_count = 10000 // mnemonic_threads
            for i in range(mnemonic_threads):
                future = executor.submit(
                    self.mnemonic_search_worker, 
                    i + sequential_threads + random_threads, 
                    mnemonic_count
                )
                futures.append(future)
            
            # Мониторинг прогресса
            try:
                while time.time() < end_time and not self.stop_flag:
                    self.print_stats()
                    time.sleep(5)
                    
                    # Сохраняем прогресс
                    if self.stats['processed_keys'] % self.config.save_interval == 0:
                        self._save_progress()
                        
            except KeyboardInterrupt:
                console.print("\n⏹️ Остановка по запросу пользователя...", style="bold red")
                self.stop_flag = True
            
            # Ждем завершения потоков
            for future in as_completed(futures, timeout=60):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Ошибка потока: {e}")
    
    def _save_progress(self):
        """Сохранение прогресса"""
        try:
            progress_data = {
                'stats': self.stats,
                'active_puzzle': self.active_puzzle.name,
                'config': {
                    'threads': self.config.threads,
                    'batch_size': self.config.batch_size,
                    'target_puzzle': self.config.target_puzzle
                },
                'timestamp': time.time()
            }
            
            with open("solver_progress.json", 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения прогресса: {e}")
    
    def run(self):
        """Главный метод запуска"""
        try:
            if self.config.search_mode == "mixed":
                self.run_mixed_search()
            else:
                console.print(f"❌ Режим '{self.config.search_mode}' пока не реализован", style="bold red")
                
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            
        finally:
            # Финальная статистика
            console.print("\n" + "="*60, style="bold")
            console.print("🏁 ФИНАЛЬНАЯ СТАТИСТИКА", style="bold green")
            console.print("="*60, style="bold")
            
            self.print_stats()
            self._save_results_to_file()
            self._save_progress()
            
            if self.found_wallets:
                console.print(f"\n💰 Найдено кошельков: {len(self.found_wallets)}")
                console.print("📁 Результаты сохранены в: all_found_wallets.json")
            else:
                console.print("\n🔍 Кошельки с балансами не найдены")

def main():
    """Главная функция"""
    console.print("🚀 Интегрированный Bitcoin Puzzle Solver v2.1", style="bold blue")
    console.print("=" * 60, style="bold")
    
    # Конфигурация
    config = SolverConfig(
        threads=8,
        batch_size=1000,
        balance_check_interval=500,
        target_puzzle="71",
        search_mode="mixed",
        duration_hours=24.0
    )
    
    # Запуск решателя
    solver = IntegratedPuzzleSolver(config)
    solver.run()

if __name__ == "__main__":
    main()
