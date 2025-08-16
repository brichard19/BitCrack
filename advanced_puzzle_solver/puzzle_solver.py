#!/usr/bin/env python3
"""
Продвинутый решатель Bitcoin Puzzle с полным функционалом:
- Генерация мнемоник и проверка кошельков
- Проверка балансов в реальном времени
- Валидация адресов
- Поддержка множественных сетей (Bitcoin, Ethereum, и др.)
- Многопоточность и оптимизация
- Автосохранение результатов
"""

import os
import sys
import json
import time
import hashlib
import secrets
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

# Криптографические библиотеки
try:
    import ecdsa
    from mnemonic import Mnemonic
    import bip32
    from bitcoin import *
    from eth_account import Account
    import base58
    from Crypto.Hash import RIPEMD160, SHA256
except ImportError as e:
    print(f"❌ Требуется установка зависимостей: pip install {e.name}")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('puzzle_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class WalletInfo:
    """Информация о кошельке"""
    private_key: str
    public_key: str
    address: str
    mnemonic: Optional[str] = None
    balance: float = 0.0
    network: str = "bitcoin"
    derivation_path: Optional[str] = None
    compressed: bool = True

@dataclass
class PuzzleConfig:
    """Конфигурация решателя"""
    # Диапазоны поиска
    start_range: int = 0x40000000000000000  # Puzzle #71
    end_range: int = 0x7FFFFFFFFFFFFFFFFF
    
    # Настройки производительности
    threads: int = 4
    batch_size: int = 1000
    check_interval: int = 100  # Проверять баланс каждые N ключей
    
    # Сети для проверки
    networks: List[str] = None
    
    # API ключи
    btc_api_key: Optional[str] = None
    eth_api_key: Optional[str] = None
    
    # Целевые адреса (если известны)
    target_addresses: List[str] = None
    
    # Файлы результатов
    results_file: str = "found_wallets.json"
    progress_file: str = "solver_progress.json"
    
    def __post_init__(self):
        if self.networks is None:
            self.networks = ["bitcoin", "ethereum"]
        if self.target_addresses is None:
            self.target_addresses = []

class NetworkChecker:
    """Проверяет балансы в различных сетях"""
    
    def __init__(self, config: PuzzleConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PuzzleSolver/1.0)'
        })
        
    def check_bitcoin_balance(self, address: str) -> float:
        """Проверка баланса Bitcoin"""
        try:
            # Используем несколько API для надежности
            apis = [
                f"https://blockstream.info/api/address/{address}",
                f"https://api.blockcypher.com/v1/btc/main/addrs/{address}/balance"
            ]
            
            for api_url in apis:
                try:
                    response = self.session.get(api_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'chain_stats' in data:  # blockstream.info
                            return data['chain_stats']['funded_txo_sum'] / 100000000
                        elif 'balance' in data:  # blockcypher
                            return data['balance'] / 100000000
                            
                except Exception as e:
                    logger.debug(f"Ошибка API {api_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Ошибка проверки Bitcoin баланса для {address}: {e}")
            
        return 0.0
    
    def check_ethereum_balance(self, address: str) -> float:
        """Проверка баланса Ethereum"""
        try:
            # Etherscan API
            if self.config.eth_api_key:
                url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={self.config.eth_api_key}"
            else:
                url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest"
                
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == '1':
                    return int(data['result']) / 10**18
                    
        except Exception as e:
            logger.error(f"Ошибка проверки Ethereum баланса для {address}: {e}")
            
        return 0.0
    
    def check_balance(self, address: str, network: str) -> float:
        """Универсальная проверка баланса"""
        if network == "bitcoin":
            return self.check_bitcoin_balance(address)
        elif network == "ethereum":
            return self.check_ethereum_balance(address)
        else:
            logger.warning(f"Неподдерживаемая сеть: {network}")
            return 0.0

class AddressValidator:
    """Валидация адресов различных криптовалют"""
    
    @staticmethod
    def validate_bitcoin_address(address: str) -> bool:
        """Валидация Bitcoin адреса"""
        try:
            if address.startswith(('1', '3')):  # P2PKH, P2SH
                decoded = base58.b58decode(address)
                if len(decoded) != 25:
                    return False
                checksum = decoded[-4:]
                payload = decoded[:-4]
                hash_result = hashlib.sha256(hashlib.sha256(payload).digest()).digest()
                return checksum == hash_result[:4]
            elif address.startswith('bc1'):  # Bech32
                return len(address) >= 42 and len(address) <= 62
        except:
            pass
        return False
    
    @staticmethod
    def validate_ethereum_address(address: str) -> bool:
        """Валидация Ethereum адреса"""
        try:
            if address.startswith('0x') and len(address) == 42:
                int(address[2:], 16)
                return True
        except:
            pass
        return False
    
    @classmethod
    def validate_address(cls, address: str, network: str) -> bool:
        """Универсальная валидация"""
        if network == "bitcoin":
            return cls.validate_bitcoin_address(address)
        elif network == "ethereum":
            return cls.validate_ethereum_address(address)
        return False

class KeyGenerator:
    """Генератор ключей и адресов"""
    
    def __init__(self):
        self.mnemonic_generator = Mnemonic("english")
    
    def private_key_to_wif(self, private_key_hex: str, compressed: bool = True) -> str:
        """Конвертация приватного ключа в WIF формат"""
        private_key_bytes = bytes.fromhex(private_key_hex)
        if compressed:
            private_key_bytes += b'\x01'
        
        # Добавляем префикс для mainnet
        extended = b'\x80' + private_key_bytes
        
        # Двойной SHA256 для чексуммы
        checksum = hashlib.sha256(hashlib.sha256(extended).digest()).digest()[:4]
        
        return base58.b58encode(extended + checksum).decode('utf-8')
    
    def private_key_to_public_key(self, private_key_hex: str, compressed: bool = True) -> str:
        """Получение публичного ключа из приватного"""
        private_key_int = int(private_key_hex, 16)
        
        # ECDSA secp256k1
        sk = ecdsa.SigningKey.from_string(
            private_key_int.to_bytes(32, 'big'),
            curve=ecdsa.SECP256k1
        )
        
        vk = sk.get_verifying_key()
        
        if compressed:
            # Сжатый формат
            x = vk.pubkey.point.x()
            y = vk.pubkey.point.y()
            prefix = b'\x02' if y % 2 == 0 else b'\x03'
            return (prefix + x.to_bytes(32, 'big')).hex()
        else:
            # Несжатый формат
            return b'\x04' + vk.to_string().hex()
    
    def public_key_to_bitcoin_address(self, public_key_hex: str) -> str:
        """Генерация Bitcoin адреса из публичного ключа"""
        public_key_bytes = bytes.fromhex(public_key_hex)
        
        # SHA256
        sha256_hash = hashlib.sha256(public_key_bytes).digest()
        
        # RIPEMD160
        ripemd160_hash = RIPEMD160.new(sha256_hash).digest()
        
        # Добавляем версию (0x00 для mainnet P2PKH)
        versioned_hash = b'\x00' + ripemd160_hash
        
        # Двойной SHA256 для чексуммы
        checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
        
        # Финальный адрес
        return base58.b58encode(versioned_hash + checksum).decode('utf-8')
    
    def private_key_to_ethereum_address(self, private_key_hex: str) -> str:
        """Генерация Ethereum адреса из приватного ключа"""
        try:
            account = Account.from_key(private_key_hex)
            return account.address
        except Exception as e:
            logger.error(f"Ошибка генерации Ethereum адреса: {e}")
            return ""
    
    def generate_mnemonic(self) -> str:
        """Генерация мнемонической фразы"""
        return self.mnemonic_generator.generate(strength=256)
    
    def mnemonic_to_private_key(self, mnemonic: str, derivation_path: str = "m/44'/0'/0'/0/0") -> str:
        """Получение приватного ключа из мнемоники"""
        try:
            seed = self.mnemonic_generator.to_seed(mnemonic)
            master_key = bip32.BIP32.from_seed(seed)
            private_key = master_key.get_privkey_from_path(derivation_path)
            return private_key.hex()
        except Exception as e:
            logger.error(f"Ошибка конвертации мнемоники: {e}")
            return ""

class PuzzleSolver:
    """Основной класс решателя"""
    
    def __init__(self, config: PuzzleConfig):
        self.config = config
        self.key_generator = KeyGenerator()
        self.network_checker = NetworkChecker(config)
        self.validator = AddressValidator()
        
        self.found_wallets = []
        self.processed_keys = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Загружаем предыдущий прогресс
        self.load_progress()
        
    def load_progress(self):
        """Загрузка прогресса из файла"""
        try:
            if os.path.exists(self.config.progress_file):
                with open(self.config.progress_file, 'r') as f:
                    data = json.load(f)
                    self.processed_keys = data.get('processed_keys', 0)
                    logger.info(f"Загружен прогресс: {self.processed_keys} обработанных ключей")
        except Exception as e:
            logger.error(f"Ошибка загрузки прогресса: {e}")
    
    def save_progress(self):
        """Сохранение прогресса"""
        try:
            progress_data = {
                'processed_keys': self.processed_keys,
                'start_time': self.start_time,
                'last_update': time.time()
            }
            with open(self.config.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения прогресса: {e}")
    
    def save_found_wallet(self, wallet: WalletInfo):
        """Сохранение найденного кошелька"""
        with self.lock:
            self.found_wallets.append(wallet)
            
            try:
                # Загружаем существующие результаты
                existing_wallets = []
                if os.path.exists(self.config.results_file):
                    with open(self.config.results_file, 'r') as f:
                        existing_wallets = json.load(f)
                
                # Добавляем новый кошелек
                existing_wallets.append(asdict(wallet))
                
                # Сохраняем
                with open(self.config.results_file, 'w') as f:
                    json.dump(existing_wallets, f, indent=2)
                    
                logger.info(f"💰 НАЙДЕН КОШЕЛЕК! {wallet.network} - {wallet.address} - Баланс: {wallet.balance}")
                
            except Exception as e:
                logger.error(f"Ошибка сохранения кошелька: {e}")
    
    def process_private_key(self, private_key_hex: str) -> List[WalletInfo]:
        """Обработка одного приватного ключа"""
        wallets = []
        
        try:
            # Генерируем адреса для разных сетей
            for network in self.config.networks:
                if network == "bitcoin":
                    # Bitcoin с сжатым и несжатым ключом
                    for compressed in [True, False]:
                        public_key = self.key_generator.private_key_to_public_key(private_key_hex, compressed)
                        address = self.key_generator.public_key_to_bitcoin_address(public_key)
                        
                        wallet = WalletInfo(
                            private_key=private_key_hex,
                            public_key=public_key,
                            address=address,
                            network=network,
                            compressed=compressed
                        )
                        
                        # Проверяем баланс
                        if self.processed_keys % self.config.check_interval == 0:
                            wallet.balance = self.network_checker.check_balance(address, network)
                        
                        wallets.append(wallet)
                        
                elif network == "ethereum":
                    address = self.key_generator.private_key_to_ethereum_address(private_key_hex)
                    if address:
                        wallet = WalletInfo(
                            private_key=private_key_hex,
                            public_key="",
                            address=address,
                            network=network
                        )
                        
                        # Проверяем баланс
                        if self.processed_keys % self.config.check_interval == 0:
                            wallet.balance = self.network_checker.check_balance(address, network)
                        
                        wallets.append(wallet)
            
            # Проверяем целевые адреса
            for wallet in wallets:
                if wallet.address in self.config.target_addresses or wallet.balance > 0:
                    self.save_found_wallet(wallet)
                    
        except Exception as e:
            logger.error(f"Ошибка обработки ключа {private_key_hex}: {e}")
        
        return wallets
    
    def generate_mnemonic_wallet(self) -> Optional[WalletInfo]:
        """Генерация кошелька из случайной мнемоники"""
        try:
            mnemonic = self.key_generator.generate_mnemonic()
            private_key_hex = self.key_generator.mnemonic_to_private_key(mnemonic)
            
            if private_key_hex:
                # Генерируем Bitcoin адрес
                public_key = self.key_generator.private_key_to_public_key(private_key_hex, True)
                bitcoin_address = self.key_generator.public_key_to_bitcoin_address(public_key)
                
                wallet = WalletInfo(
                    private_key=private_key_hex,
                    public_key=public_key,
                    address=bitcoin_address,
                    mnemonic=mnemonic,
                    network="bitcoin",
                    derivation_path="m/44'/0'/0'/0/0"
                )
                
                # Проверяем баланс
                wallet.balance = self.network_checker.check_balance(bitcoin_address, "bitcoin")
                
                return wallet
                
        except Exception as e:
            logger.error(f"Ошибка генерации мнемоники: {e}")
        
        return None
    
    def worker_thread(self, start_key: int, end_key: int, thread_id: int):
        """Рабочий поток для обработки диапазона ключей"""
        logger.info(f"Поток {thread_id} начал работу: {hex(start_key)} - {hex(end_key)}")
        
        current_key = start_key
        while current_key <= end_key:
            try:
                # Обрабатываем батч ключей
                batch_end = min(current_key + self.config.batch_size, end_key + 1)
                
                for key_int in range(current_key, batch_end):
                    private_key_hex = f"{key_int:064x}"
                    self.process_private_key(private_key_hex)
                    
                    with self.lock:
                        self.processed_keys += 1
                        
                        # Периодически сохраняем прогресс и выводим статистику
                        if self.processed_keys % 10000 == 0:
                            self.save_progress()
                            elapsed = time.time() - self.start_time
                            rate = self.processed_keys / elapsed
                            logger.info(f"Обработано ключей: {self.processed_keys}, Скорость: {rate:.2f} ключей/сек")
                
                current_key = batch_end
                
            except Exception as e:
                logger.error(f"Ошибка в потоке {thread_id}: {e}")
                current_key += self.config.batch_size
    
    def run_sequential_search(self):
        """Последовательный поиск в заданном диапазоне"""
        logger.info(f"🚀 Запуск поиска в диапазоне: {hex(self.config.start_range)} - {hex(self.config.end_range)}")
        
        # Разделяем диапазон между потоками
        total_range = self.config.end_range - self.config.start_range
        range_per_thread = total_range // self.config.threads
        
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            for i in range(self.config.threads):
                start_key = self.config.start_range + i * range_per_thread
                end_key = start_key + range_per_thread - 1
                
                if i == self.config.threads - 1:  # Последний поток берет остаток
                    end_key = self.config.end_range
                
                future = executor.submit(self.worker_thread, start_key, end_key, i)
                futures.append(future)
            
            # Ждем завершения всех потоков
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Ошибка выполнения потока: {e}")
    
    def run_random_search(self, duration_hours: int = 24):
        """Случайный поиск в течение заданного времени"""
        logger.info(f"🎲 Запуск случайного поиска на {duration_hours} часов")
        
        end_time = time.time() + duration_hours * 3600
        
        def random_worker(thread_id: int):
            while time.time() < end_time:
                try:
                    # Генерируем случайный ключ в диапазоне
                    random_key = secrets.randbelow(self.config.end_range - self.config.start_range) + self.config.start_range
                    private_key_hex = f"{random_key:064x}"
                    
                    self.process_private_key(private_key_hex)
                    
                    with self.lock:
                        self.processed_keys += 1
                        
                        if self.processed_keys % 1000 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.processed_keys / elapsed
                            remaining_time = (end_time - time.time()) / 3600
                            logger.info(f"Поток {thread_id}: {self.processed_keys} ключей, {rate:.2f} ключей/сек, осталось {remaining_time:.1f}ч")
                    
                except Exception as e:
                    logger.error(f"Ошибка в случайном поиске поток {thread_id}: {e}")
        
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            futures = [executor.submit(random_worker, i) for i in range(self.config.threads)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Ошибка случайного поиска: {e}")
    
    def run_mnemonic_search(self, count: int = 100000):
        """Поиск через генерацию случайных мнемоник"""
        logger.info(f"🔤 Запуск поиска через мнемоники: {count} попыток")
        
        def mnemonic_worker(thread_id: int, attempts: int):
            for i in range(attempts):
                try:
                    wallet = self.generate_mnemonic_wallet()
                    if wallet:
                        if wallet.balance > 0 or wallet.address in self.config.target_addresses:
                            self.save_found_wallet(wallet)
                    
                    with self.lock:
                        self.processed_keys += 1
                        
                        if self.processed_keys % 1000 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.processed_keys / elapsed
                            logger.info(f"Мнемоники: {self.processed_keys}, {rate:.2f}/сек")
                
                except Exception as e:
                    logger.error(f"Ошибка мнемоники поток {thread_id}: {e}")
        
        attempts_per_thread = count // self.config.threads
        
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            futures = [executor.submit(mnemonic_worker, i, attempts_per_thread) for i in range(self.config.threads)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Ошибка поиска мнемоник: {e}")

def main():
    """Основная функция"""
    print("🔍 Продвинутый Bitcoin Puzzle Solver")
    print("=====================================")
    
    # Конфигурация для Puzzle #71
    config = PuzzleConfig(
        start_range=0x40000000000000000,  # 2^70
        end_range=0x7FFFFFFFFFFFFFFFFF,   # 2^71 - 1
        threads=8,  # Увеличиваем количество потоков
        batch_size=1000,
        check_interval=100,
        networks=["bitcoin", "ethereum"],
        target_addresses=[
            # Добавьте сюда известные адреса с балансами если есть
        ]
    )
    
    solver = PuzzleSolver(config)
    
    try:
        print("\nВыберите режим поиска:")
        print("1. Последовательный поиск в диапазоне")
        print("2. Случайный поиск (24 часа)")
        print("3. Поиск через мнемоники")
        print("4. Смешанный режим")
        
        choice = input("Введите номер (1-4): ").strip()
        
        if choice == "1":
            solver.run_sequential_search()
        elif choice == "2":
            hours = int(input("Количество часов для поиска (по умолчанию 24): ") or 24)
            solver.run_random_search(hours)
        elif choice == "3":
            count = int(input("Количество мнемоник для генерации (по умолчанию 100000): ") or 100000)
            solver.run_mnemonic_search(count)
        elif choice == "4":
            # Смешанный режим: немного всего
            print("🔀 Смешанный режим: последовательный + случайный + мнемоники")
            
            # 30 минут случайного поиска
            threading.Thread(target=solver.run_random_search, args=(0.5,)).start()
            
            # 10000 мнемоник
            threading.Thread(target=solver.run_mnemonic_search, args=(10000,)).start()
            
            # Основной поток - последовательный поиск
            solver.run_sequential_search()
        else:
            print("❌ Неверный выбор")
            return
    
    except KeyboardInterrupt:
        print("\n⏹️ Поиск остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        solver.save_progress()
        print(f"\n📊 Итоговая статистика:")
        print(f"Обработано ключей: {solver.processed_keys}")
        print(f"Найдено кошельков: {len(solver.found_wallets)}")
        elapsed = time.time() - solver.start_time
        if elapsed > 0:
            print(f"Средняя скорость: {solver.processed_keys / elapsed:.2f} ключей/сек")
        
        if solver.found_wallets:
            print(f"\n💰 Найденные кошельки сохранены в: {config.results_file}")

if __name__ == "__main__":
    main()
