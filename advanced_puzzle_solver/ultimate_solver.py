#!/usr/bin/env python3
"""
🔥 ULTIMATE BITCOIN PUZZLE SOLVER v3.0 🔥
Максимальная производительность с профилированием и валидацией
Разработано как лучший серый хак всех времен
"""

import hashlib
import multiprocessing as mp
import threading
import time
import secrets
import logging
import json
import os
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ecdsa
from ecdsa.ellipticcurve import Point
from ecdsa.curves import SECP256k1
import base58
import psutil
import cProfile
import pstats
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import asyncio
import aiohttp
import queue
from contextlib import asynccontextmanager

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceStats:
    """Статистика производительности"""
    keys_per_second: float = 0.0
    total_keys_checked: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    thread_efficiency: float = 0.0
    api_success_rate: float = 0.0
    uptime: float = 0.0

class UltimateProfiler:
    """Профайлер производительности"""
    
    def __init__(self):
        self.stats = PerformanceStats()
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.last_keys_count = 0
        self.api_calls = {'success': 0, 'failed': 0}
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        
    def start_profiling(self):
        """Начать профилирование"""
        self.profiler.enable()
        self.is_profiling = True
        logger.info("🔍 Профилирование ВКЛЮЧЕНО")
        
    def stop_profiling(self):
        """Остановить профилирование и сохранить результаты"""
        if self.is_profiling:
            self.profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Топ 20 функций
            
            with open('performance_profile.txt', 'w') as f:
                f.write(s.getvalue())
            
            logger.info("📊 Профиль производительности сохранен в performance_profile.txt")
            self.is_profiling = False
    
    def update_stats(self, keys_checked: int):
        """Обновить статистику"""
        current_time = time.time()
        
        # Скорость ключей в секунду
        time_diff = current_time - self.last_check_time
        if time_diff > 0:
            keys_diff = keys_checked - self.last_keys_count
            self.stats.keys_per_second = keys_diff / time_diff
            
        self.stats.total_keys_checked = keys_checked
        self.stats.cpu_usage = psutil.cpu_percent(interval=None)
        self.stats.memory_usage = psutil.virtual_memory().percent
        self.stats.uptime = current_time - self.start_time
        
        # API success rate
        total_api = self.api_calls['success'] + self.api_calls['failed']
        if total_api > 0:
            self.stats.api_success_rate = self.api_calls['success'] / total_api * 100
            
        self.last_check_time = current_time
        self.last_keys_count = keys_checked
        
    def log_performance(self):
        """Вывести текущую статистику производительности"""
        logger.info(f"""
🚀 === СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ ===
⚡ Скорость: {self.stats.keys_per_second:,.0f} ключей/сек
🔢 Всего проверено: {self.stats.total_keys_checked:,}
💽 Использование CPU: {self.stats.cpu_usage:.1f}%
🧠 Использование RAM: {self.stats.memory_usage:.1f}%
🌐 API Success Rate: {self.stats.api_success_rate:.1f}%
⏱️ Время работы: {self.stats.uptime/3600:.2f} часов
==========================================
""")

class UltimateValidator:
    """Максимально быстрый валидатор адресов"""
    
    def __init__(self):
        # Кеш для ускорения валидации
        self.address_cache: Dict[str, bool] = {}
        self.point_cache: Dict[int, Point] = {}
        
        # Предвычисленные константы
        self.secp256k1_p = SECP256k1.generator.curve().p()
        self.secp256k1_n = SECP256k1.generator.order()
        self.secp256k1_g = SECP256k1.generator
        
        # Batch валидация для API
        self.batch_queue = queue.Queue(maxsize=100)
        self.batch_size = 20
        
        logger.info("✅ Ultimate Validator инициализирован")
        
    def private_key_to_point(self, private_key: int) -> Point:
        """Максимально быстрое преобразование приватного ключа в точку"""
        if private_key in self.point_cache:
            return self.point_cache[private_key]
            
        # Используем быстрое умножение на эллиптической кривой
        point = self.secp256k1_g * private_key
        
        # Кешируем только важные точки
        if len(self.point_cache) < 10000:
            self.point_cache[private_key] = point
            
        return point
    
    def point_to_address(self, point: Point, compressed: bool = True) -> str:
        """Максимально быстрое преобразование точки в адрес"""
        try:
            x_bytes = point.x().to_bytes(32, 'big')
            y_bytes = point.y().to_bytes(32, 'big')
            
            if compressed:
                if point.y() % 2 == 0:
                    public_key = b'\x02' + x_bytes
                else:
                    public_key = b'\x03' + x_bytes
            else:
                public_key = b'\x04' + x_bytes + y_bytes
            
            # SHA256 + RIPEMD160
            sha256_hash = hashlib.sha256(public_key).digest()
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha256_hash)
            hash160 = ripemd160.digest()
            
            # Base58Check
            versioned_hash = b'\x00' + hash160
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            return base58.b58encode(versioned_hash + checksum).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Ошибка преобразования точки в адрес: {e}")
            return ""
    
    def is_valid_private_key(self, key: int) -> bool:
        """Быстрая проверка валидности приватного ключа"""
        return 1 <= key < self.secp256k1_n
    
    def batch_validate_addresses(self, addresses: List[str]) -> List[bool]:
        """Батч валидация адресов"""
        results = []
        for addr in addresses:
            if addr in self.address_cache:
                results.append(self.address_cache[addr])
            else:
                is_valid = self._validate_address_format(addr)
                self.address_cache[addr] = is_valid
                results.append(is_valid)
                
                # Ограничиваем размер кеша
                if len(self.address_cache) > 50000:
                    # Удаляем 25% старых записей
                    items_to_remove = list(self.address_cache.keys())[:12500]
                    for key in items_to_remove:
                        del self.address_cache[key]
                        
        return results
    
    def _validate_address_format(self, address: str) -> bool:
        """Быстрая проверка формата Bitcoin адреса"""
        try:
            if not address or len(address) < 26 or len(address) > 35:
                return False
                
            if not address.startswith('1'):
                return False
                
            # Base58 декодирование с проверкой checksum
            decoded = base58.b58decode(address)
            if len(decoded) != 25:
                return False
                
            checksum = decoded[-4:]
            payload = decoded[:-4]
            hash_result = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            
            return checksum == hash_result
            
        except Exception:
            return False

class UltimateBalanceChecker:
    """Максимально быстрый проверщик балансов с множественными API"""
    
    def __init__(self):
        self.session = self._create_session()
        self.api_endpoints = [
            {
                'name': 'BlockCypher',
                'url': 'https://api.blockcypher.com/v1/btc/main/addrs/{}/balance',
                'timeout': 3,
                'rate_limit': 0.1  # 10 requests per second
            },
            {
                'name': 'BlockStream',
                'url': 'https://blockstream.info/api/address/{}',
                'timeout': 3,
                'rate_limit': 0.2  # 5 requests per second
            },
            {
                'name': 'Chain.so',
                'url': 'https://chain.so/api/v2/get_address_balance/BTC/{}',
                'timeout': 3,
                'rate_limit': 0.15
            }
        ]
        
        self.current_api = 0
        self.last_api_call = 0
        self.batch_queue = asyncio.Queue(maxsize=1000)
        
        logger.info("💰 Ultimate Balance Checker инициализирован")
    
    def _create_session(self) -> requests.Session:
        """Создать оптимизированную сессию"""
        session = requests.Session()
        
        # Настройки повторных попыток
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def check_balance_fast(self, address: str) -> Tuple[bool, float]:
        """Быстрая проверка баланса"""
        current_time = time.time()
        api = self.api_endpoints[self.current_api]
        
        # Rate limiting
        time_since_last = current_time - self.last_api_call
        if time_since_last < api['rate_limit']:
            time.sleep(api['rate_limit'] - time_since_last)
        
        try:
            url = api['url'].format(address)
            response = self.session.get(url, timeout=api['timeout'])
            
            if response.status_code == 200:
                balance = self._parse_balance_response(response.json(), api['name'])
                self.last_api_call = time.time()
                return True, balance
            else:
                self._switch_api()
                return False, 0.0
                
        except Exception as e:
            logger.debug(f"API error ({api['name']}): {str(e)}")
            self._switch_api()
            return False, 0.0
    
    def _parse_balance_response(self, data: dict, api_name: str) -> float:
        """Парсинг ответа API"""
        try:
            if api_name == 'BlockCypher':
                return float(data.get('balance', 0)) / 100000000  # satoshi to BTC
            elif api_name == 'BlockStream':
                confirmed = data.get('chain_stats', {}).get('funded_txo_sum', 0)
                spent = data.get('chain_stats', {}).get('spent_txo_sum', 0)
                return float(confirmed - spent) / 100000000
            elif api_name == 'Chain.so':
                return float(data.get('data', {}).get('confirmed_balance', 0))
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Ошибка парсинга ответа {api_name}: {e}")
            return 0.0
    
    def _switch_api(self):
        """Переключить на следующий API"""
        self.current_api = (self.current_api + 1) % len(self.api_endpoints)
        logger.debug(f"Переключение на API: {self.api_endpoints[self.current_api]['name']}")

class UltimatePuzzleSolver:
    """Максимально производительный решатель puzzle"""
    
    def __init__(self):
        self.is_running = False
        self.profiler = UltimateProfiler()
        self.validator = UltimateValidator()
        self.balance_checker = UltimateBalanceChecker()
        
        # Puzzle диапазоны
        self.puzzle_ranges = {
            63: (0x4000000000000000, 0x7FFFFFFFFFFFFFFF),
            64: (0x8000000000000000, 0xFFFFFFFFFFFFFFFF),
            66: (0x20000000000000000, 0x3FFFFFFFFFFFFFFFF)
        }
        
        # Известные адреса с балансами
        self.target_addresses = self._load_targets()
        
        # Статистика
        self.total_keys_checked = mp.Value('Q', 0)  # Unsigned long long
        self.keys_found = mp.Value('i', 0)
        self.start_time = time.time()
        
        # Настройки производительности
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.optimal_threads = min(self.cpu_cores * 2, 40)  # Не более 40 потоков
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🔥 Ultimate Puzzle Solver v3.0 загружен")
        logger.info(f"💻 Обнаружено {self.cpu_cores} CPU ядер")
        logger.info(f"🧵 Будет использовано {self.optimal_threads} потоков")
        
    def _load_targets(self) -> Set[str]:
        """Загрузить целевые адреса"""
        targets = set()
        
        # Puzzle адреса
        puzzle_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block
            "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",  # Puzzle #60
            "1CUhaTe2wecqRuqPiqueXstmK6P4Lkoz3J",  # Puzzle #61
            "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1",  # Puzzle #62
            "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",  # Puzzle #63
            "1QFqqMUD55ZV3PJEJZtaKCsQmjLT6JkjvJ",  # Puzzle #64
            "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",  # Puzzle #66
        ]
        
        for addr in puzzle_addresses:
            targets.add(addr)
            
        logger.info(f"📍 Загружено {len(targets)} целевых адресов")
        return targets
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов остановки"""
        logger.warning("🛑 Получен сигнал остановки...")
        self.is_running = False
        self.profiler.stop_profiling()
        self._save_final_report()
        sys.exit(0)
    
    def _save_final_report(self):
        """Сохранить финальный отчет"""
        report = {
            'timestamp': time.time(),
            'total_keys_checked': self.total_keys_checked.value,
            'keys_found': self.keys_found.value,
            'runtime_hours': (time.time() - self.start_time) / 3600,
            'performance_stats': {
                'keys_per_second': self.profiler.stats.keys_per_second,
                'cpu_usage': self.profiler.stats.cpu_usage,
                'memory_usage': self.profiler.stats.memory_usage,
                'api_success_rate': self.profiler.stats.api_success_rate
            }
        }
        
        with open('ultimate_solver_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info("📊 Финальный отчет сохранен в ultimate_solver_report.json")
    
    def ultra_fast_worker(self, worker_id: int, search_range: Tuple[int, int], strategy: str):
        """Ультра-быстрый воркер для поиска ключей"""
        logger.info(f"🔥 Воркер {worker_id} запущен: {strategy}")
        
        local_checked = 0
        last_report = time.time()
        
        try:
            if strategy == "sequential_aggressive":
                # Агрессивный последовательный поиск
                current_key = search_range[0] + worker_id * 10000
                step = self.optimal_threads * 10000
                
                while self.is_running and current_key < search_range[1]:
                    # Батч обработка по 1000 ключей
                    for i in range(1000):
                        if not self.is_running:
                            break
                            
                        if self.validator.is_valid_private_key(current_key):
                            point = self.validator.private_key_to_point(current_key)
                            
                            # Проверяем оба формата адресов
                            for compressed in [True, False]:
                                address = self.validator.point_to_address(point, compressed)
                                
                                if address in self.target_addresses:
                                    logger.critical(f"🎯 НАЙДЕН КЛЮЧ! Адрес: {address}, Приватный ключ: {hex(current_key)}")
                                    self._save_found_key(address, hex(current_key))
                                    with self.keys_found.get_lock():
                                        self.keys_found.value += 1
                                
                                # Проверяем баланс у случайных адресов
                                if local_checked % 50000 == 0:  # Каждый 50000-й ключ
                                    has_balance, balance = self.balance_checker.check_balance_fast(address)
                                    if has_balance and balance > 0:
                                        logger.critical(f"💰 НАЙДЕН АДРЕС С БАЛАНСОМ! {address}: {balance} BTC, Ключ: {hex(current_key)}")
                                        self._save_found_key(address, hex(current_key), balance)
                        
                        current_key += step
                        local_checked += 1
                    
                    # Обновляем общую статистику
                    with self.total_keys_checked.get_lock():
                        self.total_keys_checked.value += 1000
                    
                    # Отчет каждые 10 секунд
                    if time.time() - last_report > 10:
                        self.profiler.update_stats(self.total_keys_checked.value)
                        last_report = time.time()
                        
            elif strategy == "random_blast":
                # Случайный взрыв поиск
                while self.is_running:
                    # Батч случайных ключей
                    for _ in range(1000):
                        if not self.is_running:
                            break
                            
                        # Генерируем случайный ключ в диапазоне puzzle 63-66
                        random_bits = secrets.randbits(64)
                        if random_bits == 0:
                            continue
                            
                        point = self.validator.private_key_to_point(random_bits)
                        
                        for compressed in [True, False]:
                            address = self.validator.point_to_address(point, compressed)
                            
                            if address in self.target_addresses:
                                logger.critical(f"🎯 СЛУЧАЙНО НАЙДЕН! Адрес: {address}, Ключ: {hex(random_bits)}")
                                self._save_found_key(address, hex(random_bits))
                                with self.keys_found.get_lock():
                                    self.keys_found.value += 1
                        
                        local_checked += 1
                    
                    with self.total_keys_checked.get_lock():
                        self.total_keys_checked.value += 1000
                        
                    if time.time() - last_report > 10:
                        self.profiler.update_stats(self.total_keys_checked.value)
                        last_report = time.time()
                        
            elif strategy == "smart_pattern":
                # Умный поиск по паттернам
                while self.is_running:
                    # Генерируем ключи на основе математических паттернов
                    base_key = secrets.randbits(63) | (1 << 62)  # Puzzle 63 диапазон
                    
                    # Пробуем различные паттерны
                    patterns = [
                        base_key,
                        base_key | 0xFFFF,  # Установить младшие биты
                        base_key & ~0xFFFF,  # Сбросить младшие биты
                        base_key ^ 0xAAAAAAAAAAAAAAAA,  # XOR с паттерном
                        base_key + 1,
                        base_key - 1,
                        base_key * 2 % (1 << 63),
                        base_key >> 1,
                    ]
                    
                    for pattern_key in patterns:
                        if not self.is_running:
                            break
                            
                        if not self.validator.is_valid_private_key(pattern_key):
                            continue
                            
                        point = self.validator.private_key_to_point(pattern_key)
                        
                        for compressed in [True, False]:
                            address = self.validator.point_to_address(point, compressed)
                            
                            if address in self.target_addresses:
                                logger.critical(f"🎯 ПАТТЕРН НАЙДЕН! Адрес: {address}, Ключ: {hex(pattern_key)}")
                                self._save_found_key(address, hex(pattern_key))
                                with self.keys_found.get_lock():
                                    self.keys_found.value += 1
                        
                        local_checked += 1
                    
                    with self.total_keys_checked.get_lock():
                        self.total_keys_checked.value += len(patterns)
                        
                    if time.time() - last_report > 10:
                        self.profiler.update_stats(self.total_keys_checked.value)
                        last_report = time.time()
                        
        except Exception as e:
            logger.error(f"Ошибка в воркере {worker_id}: {e}")
            traceback.print_exc()
        
        logger.info(f"🔚 Воркер {worker_id} завершен: проверено {local_checked:,} ключей")
    
    def _save_found_key(self, address: str, private_key: str, balance: float = None):
        """Сохранить найденный ключ"""
        result = {
            'timestamp': time.time(),
            'address': address,
            'private_key': private_key,
            'balance_btc': balance,
            'compressed': True  # Можно определить по адресу
        }
        
        # Сохранить в JSON
        filename = f'FOUND_KEY_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Также добавить в общий лог
        with open('all_found_keys.txt', 'a') as f:
            f.write(f"{time.ctime()}: {address} -> {private_key}\n")
            
        logger.critical(f"💾 Результат сохранен в {filename}")
    
    def start_ultimate_search(self):
        """Запустить максимально производительный поиск"""
        logger.info("🚀 ЗАПУСК МАКСИМАЛЬНОГО ПОИСКА 🚀")
        
        self.is_running = True
        self.profiler.start_profiling()
        
        # Создаем пул воркеров
        workers = []
        
        # 60% потоков для агрессивного последовательного поиска
        sequential_threads = int(self.optimal_threads * 0.6)
        puzzle_63_range = self.puzzle_ranges[63]
        range_size = (puzzle_63_range[1] - puzzle_63_range[0]) // sequential_threads
        
        for i in range(sequential_threads):
            start = puzzle_63_range[0] + i * range_size
            end = start + range_size if i < sequential_threads - 1 else puzzle_63_range[1]
            
            worker = threading.Thread(
                target=self.ultra_fast_worker,
                args=(i, (start, end), "sequential_aggressive"),
                daemon=True
            )
            workers.append(worker)
            worker.start()
        
        # 25% потоков для случайного поиска
        random_threads = int(self.optimal_threads * 0.25)
        for i in range(random_threads):
            worker = threading.Thread(
                target=self.ultra_fast_worker,
                args=(sequential_threads + i, (0, 0), "random_blast"),
                daemon=True
            )
            workers.append(worker)
            worker.start()
        
        # 15% потоков для умного поиска
        smart_threads = self.optimal_threads - sequential_threads - random_threads
        for i in range(smart_threads):
            worker = threading.Thread(
                target=self.ultra_fast_worker,
                args=(sequential_threads + random_threads + i, (0, 0), "smart_pattern"),
                daemon=True
            )
            workers.append(worker)
            worker.start()
        
        logger.info(f"🧵 Запущено {len(workers)} воркеров:")
        logger.info(f"   • {sequential_threads} агрессивных последовательных")
        logger.info(f"   • {random_threads} случайных взрывных") 
        logger.info(f"   • {smart_threads} умных паттерн-поисковиков")
        
        # Главный цикл мониторинга
        try:
            while self.is_running:
                time.sleep(30)  # Отчет каждые 30 секунд
                
                self.profiler.update_stats(self.total_keys_checked.value)
                self.profiler.log_performance()
                
                # Проверяем "живость" потоков
                alive_workers = sum(1 for w in workers if w.is_alive())
                logger.info(f"👥 Активных воркеров: {alive_workers}/{len(workers)}")
                
                # Если упало много потоков, перезапускаем
                if alive_workers < len(workers) * 0.8:
                    logger.warning("🔄 Обнаружено падение потоков, перезапускаем...")
                    break
                    
        except KeyboardInterrupt:
            logger.warning("🛑 Получен сигнал остановки от пользователя")
            self.is_running = False
        
        # Ожидание завершения потоков
        logger.info("⏳ Ожидание завершения воркеров...")
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=5)
        
        self.profiler.stop_profiling()
        self._save_final_report()
        
        logger.info("🏁 Ultimate Solver завершен!")
        logger.info(f"📊 Итого проверено: {self.total_keys_checked.value:,} ключей")
        logger.info(f"🎯 Найдено ключей: {self.keys_found.value}")

def main():
    """Главная функция"""
    print("🔥" * 50)
    print("🔥 ULTIMATE BITCOIN PUZZLE SOLVER v3.0 🔥")
    print("🔥 Максимальная производительность и валидация 🔥")
    print("🔥" * 50)
    
    try:
        solver = UltimatePuzzleSolver()
        solver.start_ultimate_search()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
