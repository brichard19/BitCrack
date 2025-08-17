#!/usr/bin/env python3
"""
🚀 REAL BITCOIN PUZZLE SOLVER - Настоящий высокопроизводительный решатель
Использует современные библиотеки для максимальной скорости генерации Bitcoin ключей
"""

import os
import sys
import time
import hashlib
import secrets
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import requests
import json

# Современные криптографические библиотеки
try:
    import secp256k1  # Быстрая библиотека secp256k1
    import base58
    import bitcoin  # python-bitcoinlib
    from bitcoin.core import *
    from bitcoin.wallet import *
    import coincurve  # Еще одна быстрая secp256k1
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC
except ImportError as e:
    print(f"❌ Устанавливаю зависимости: {e.name}")
    os.system(f"pip install secp256k1 coincurve python-bitcoinlib base58 hdwallet")
    print("✅ Зависимости установлены, перезапустите программу")
    sys.exit(0)

class RealBitcoinSolver:
    """Настоящий Bitcoin Puzzle Solver с максимальной производительностью"""
    
    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.total_keys_tested = mp.Value('Q', 0)  # Счетчик ключей
        self.found_keys = mp.Manager().list()  # Найденные ключи
        self.running = mp.Value('i', 1)  # Флаг работы
        self.start_time = time.time()
        
        # Bitcoin puzzle addresses (настоящие)
        self.puzzle_targets = {
            64: "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",
            65: "18ZMbwUFLMHoZBbfpCjUJQTCu352eqkeLT", 
            66: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
            67: "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9",
            68: "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",
            69: "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG",
            70: "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
        }
        
        print("🚀 REAL Bitcoin Puzzle Solver v3.0")
        print("=" * 60)
        print(f"💻 CPU cores: {self.cpu_count}")
        print(f"🎯 Target puzzles: {list(self.puzzle_targets.keys())}")
        print(f"🔥 Using modern crypto libraries: secp256k1, coincurve")
        
        # Создаем директории
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def fast_private_to_wif(self, private_key_int, compressed=True):
        """Быстрое преобразование в WIF формат"""
        # Добавляем префикс mainnet (0x80)
        extended_key = bytes([0x80]) + private_key_int.to_bytes(32, 'big')
        if compressed:
            extended_key += bytes([0x01])
        
        # Двойной SHA256 для контрольной суммы
        hash1 = hashlib.sha256(extended_key).digest()
        hash2 = hashlib.sha256(hash1).digest()
        checksum = hash2[:4]
        
        # Кодируем в Base58
        return base58.b58encode(extended_key + checksum).decode()
    
    def fast_private_to_address(self, private_key_int, compressed=True):
        """Максимально быстрое преобразование приватного ключа в Bitcoin адрес"""
        try:
            # Используем coincurve для максимальной скорости
            private_key_bytes = private_key_int.to_bytes(32, 'big')
            private_key_obj = coincurve.PrivateKey(private_key_bytes)
            
            # Получаем публичный ключ
            if compressed:
                public_key_bytes = private_key_obj.public_key.format(compressed=True)
            else:
                public_key_bytes = private_key_obj.public_key.format(compressed=False)
            
            # SHA256 -> RIPEMD160
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha256_hash)
            pub_key_hash = ripemd160.digest()
            
            # Добавляем версию (0x00 для mainnet P2PKH)
            versioned_hash = b'\x00' + pub_key_hash
            
            # Контрольная сумма
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            # Финальный адрес
            address_bytes = versioned_hash + checksum
            return base58.b58encode(address_bytes).decode()
            
        except Exception as e:
            return None
    
    def check_balance_blockstream(self, address):
        """Проверка баланса через Blockstream API"""
        try:
            url = f"https://blockstream.info/api/address/{address}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('chain_stats', {}).get('funded_txo_sum', 0) / 100000000  # Satoshi to BTC
        except:
            pass
        return 0
    
    def real_worker(self, worker_id, start_range, end_range, target_puzzle):
        """Реальный воркер для поиска Bitcoin ключей"""
        print(f"🔥 Worker {worker_id} started: puzzle {target_puzzle}")
        
        target_address = self.puzzle_targets[target_puzzle]
        local_count = 0
        last_report = time.time()
        
        current_key = start_range + worker_id  # Смещение для каждого воркера
        
        while self.running.value and current_key < end_range:
            try:
                # Генерируем адрес (сжатый)
                address_compressed = self.fast_private_to_address(current_key, compressed=True)
                
                if address_compressed == target_address:
                    print(f"\n🎉🎉🎉 НАЙДЕН КЛЮЧ PUZZLE {target_puzzle}! 🎉🎉🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private Key (hex): {hex(current_key)}")
                    print(f"Private Key (WIF): {self.fast_private_to_wif(current_key, True)}")
                    print(f"Address: {address_compressed}")
                    
                    # Проверяем баланс
                    balance = self.check_balance_blockstream(address_compressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем результат
                    result = {
                        'puzzle': target_puzzle,
                        'worker_id': worker_id,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.fast_private_to_wif(current_key, True),
                        'address': address_compressed,
                        'balance': balance,
                        'found_at': time.ctime(),
                        'compressed': True
                    }
                    
                    with open(f"results/FOUND_PUZZLE_{target_puzzle}_{worker_id}.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.found_keys.append(result)
                    return True
                
                # Тестируем несжатый адрес
                address_uncompressed = self.fast_private_to_address(current_key, compressed=False)
                
                if address_uncompressed == target_address:
                    print(f"\n🎉🎉🎉 НАЙДЕН КЛЮЧ PUZZLE {target_puzzle}! (UNCOMPRESSED) 🎉🎉🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private Key (hex): {hex(current_key)}")
                    print(f"Private Key (WIF): {self.fast_private_to_wif(current_key, False)}")
                    print(f"Address: {address_uncompressed}")
                    
                    # Проверяем баланс
                    balance = self.check_balance_blockstream(address_uncompressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем результат
                    result = {
                        'puzzle': target_puzzle,
                        'worker_id': worker_id,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.fast_private_to_wif(current_key, False),
                        'address': address_uncompressed,
                        'balance': balance,
                        'found_at': time.ctime(),
                        'compressed': False
                    }
                    
                    with open(f"results/FOUND_PUZZLE_{target_puzzle}_{worker_id}.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.found_keys.append(result)
                    return True
                
            except Exception as e:
                pass
            
            current_key += self.cpu_count  # Прыжок на количество воркеров
            local_count += 1
            
            # Обновляем статистику каждые 50000 ключей
            if local_count % 50000 == 0:
                with self.total_keys_tested.get_lock():
                    self.total_keys_tested.value += 50000
                
                # Отчет каждые 30 секунд
                if time.time() - last_report > 30:
                    elapsed = time.time() - self.start_time
                    rate = local_count / elapsed if elapsed > 0 else 0
                    print(f"⚡ Worker {worker_id}: {local_count:,} keys, {rate:.0f} k/s")
                    last_report = time.time()
        
        return False
    
    def run_real_mode(self, target_puzzle=66, max_workers=None):
        """Запуск реального поиска"""
        if max_workers is None:
            max_workers = min(self.cpu_count, 24)
        
        print(f"\n🎯 АТАКУЕМ PUZZLE {target_puzzle}")
        print(f"📍 Target address: {self.puzzle_targets[target_puzzle]}")
        print(f"👥 Workers: {max_workers}")
        print(f"🔥 Let's find some Bitcoin!")
        
        # Диапазон для поиска
        if target_puzzle == 64:
            start = 0x8000000000000000
            end = 0xFFFFFFFFFFFFFFFF
        elif target_puzzle == 65:
            start = 0x10000000000000000
            end = 0x1FFFFFFFFFFFFFFFF
        elif target_puzzle == 66:
            start = 0x20000000000000000
            end = 0x3FFFFFFFFFFFFFFFF
        else:
            start = 2**(target_puzzle-1)
            end = 2**target_puzzle - 1
        
        print(f"📊 Range: {hex(start)} - {hex(end)}")
        print("-" * 60)
        
        # Запускаем воркеры
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = (end - start) // max_workers
            
            futures = []
            for i in range(max_workers):
                worker_start = start + i * chunk_size
                worker_end = start + (i + 1) * chunk_size if i < max_workers - 1 else end
                
                future = executor.submit(self.real_worker, i, worker_start, worker_end, target_puzzle)
                futures.append(future)
            
            # Мониторинг прогресса
            last_stats = time.time()
            try:
                while True:
                    # Проверяем завершенные задачи
                    done_count = sum(1 for f in futures if f.done())
                    
                    if done_count > 0:
                        for f in futures:
                            if f.done() and f.result():
                                print("🎉 KEY FOUND! Stopping other workers...")
                                self.running.value = 0
                                return True
                    
                    if done_count == len(futures):
                        break
                    
                    # Статистика каждые 60 секунд
                    if time.time() - last_stats > 60:
                        elapsed = time.time() - self.start_time
                        total = self.total_keys_tested.value
                        rate = total / elapsed if elapsed > 0 else 0
                        
                        print(f"📈 Total: {total:,} keys, {rate:.0f} keys/sec, {elapsed:.0f}s")
                        print(f"🧵 Active workers: {max_workers - done_count}/{max_workers}")
                        
                        last_stats = time.time()
                    
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                self.running.value = 0
        
        return False

def main():
    print("🚀 Real Bitcoin Puzzle Solver - Maximum Performance")
    print("Using modern crypto libraries for real BTC network")
    print("=" * 60)
    
    solver = RealBitcoinSolver()
    
    # Выбираем puzzle для атаки (66 - хорошая цель)
    target_puzzle = 66
    
    print(f"\nStarting attack on Puzzle {target_puzzle}...")
    print(f"This will use maximum CPU power!")
    
    result = solver.run_real_mode(target_puzzle=target_puzzle)
    
    if result:
        print("\n🎉 SUCCESS! Bitcoin key found!")
    else:
        print("\n⏹️  Search session completed")
    
    print(f"Total keys tested: {solver.total_keys_tested.value:,}")
    print(f"Session time: {(time.time() - solver.start_time):.1f} seconds")

if __name__ == "__main__":
    main()
