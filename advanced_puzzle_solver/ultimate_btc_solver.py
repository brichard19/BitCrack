#!/usr/bin/env python3
"""
🏆 ULTIMATE Bitcoin Puzzle Solver
Использует самые современные библиотеки и методы валидации
"""

import os
import sys
import time
import hashlib
import secrets
import threading
import json
import multiprocessing as mp

# Проверяем и устанавливаем зависимости
try:
    # Современные криптографические библиотеки
    from coincurve import PrivateKey, PublicKey  # Самая быстрая secp256k1
    import base58
    import requests
    
    # Для валидации адресов
    from bitcoin import base58_check_decode, base58_check_encode
    from bitcoin.segwit_addr import bech32_decode, bech32_encode
    
    # Для мониторинга производительности
    import psutil
    
    print("✅ All modern libraries loaded successfully!")
    
except ImportError as e:
    print(f"❌ Missing library: {e.name}")
    print("🔧 Run: ./wsl_reset.sh to install all dependencies")
    sys.exit(1)

class UltimateBitcoinSolver:
    """Самый современный Bitcoin Puzzle Solver"""
    
    def __init__(self):
        # Системная информация
        self.cpu_count = os.cpu_count()
        self.memory_gb = psutil.virtual_memory().total // (1024**3)
        
        # Статистика с shared memory
        self.stats = mp.Manager().dict({
            'keys_tested': 0,
            'addresses_generated': 0,
            'start_time': time.time(),
            'last_report_time': time.time()
        })
        
        # Найденные результаты
        self.results = mp.Manager().list()
        
        # Флаг работы
        self.running = mp.Value('i', 1)
        
        # НАСТОЯЩИЕ Bitcoin Puzzle адреса
        self.bitcoin_puzzles = {
            64: {
                'address': '16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN',
                'range_start': 0x8000000000000000,
                'range_end': 0xFFFFFFFFFFFFFFFF,
                'reward': '6.4 BTC'
            },
            65: {
                'address': '18ZMbwUFLMHoZBbfpCjUJQTCu352eqkeLT',
                'range_start': 0x10000000000000000,
                'range_end': 0x1FFFFFFFFFFFFFFFF,
                'reward': '6.5 BTC'
            },
            66: {
                'address': '13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so',
                'range_start': 0x20000000000000000,
                'range_end': 0x3FFFFFFFFFFFFFFFF,
                'reward': '6.6 BTC'
            },
            67: {
                'address': '1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9',
                'range_start': 0x40000000000000000,
                'range_end': 0x7FFFFFFFFFFFFFFFF,
                'reward': '6.7 BTC'
            }
        }
        
        print("🏆 ULTIMATE Bitcoin Puzzle Solver v4.0")
        print("=" * 60)
        print(f"💻 System: {self.cpu_count} CPU cores, {self.memory_gb} GB RAM")
        print(f"🎯 Bitcoin Puzzles: {list(self.bitcoin_puzzles.keys())}")
        print(f"🔥 Using: coincurve (fastest secp256k1 implementation)")
        
        # Создаем директории
        os.makedirs("ultimate_results", exist_ok=True)
    
    def validate_bitcoin_address(self, address):
        """Валидация Bitcoin адреса всех типов"""
        try:
            # P2PKH и P2SH (Base58Check)
            if address.startswith(('1', '3')):
                decoded = base58_check_decode(address)
                return len(decoded) == 20  # 20 bytes for hash160
            
            # Bech32 (Segwit)
            elif address.startswith('bc1'):
                hrp, data = bech32_decode(address)
                return hrp == 'bc' and data is not None
                
        except Exception:
            pass
        
        return False
    
    def private_key_to_address(self, private_key_int, compressed=True, address_type='p2pkh'):
        """Современное преобразование приватного ключа в Bitcoin адрес"""
        try:
            # Создаем приватный ключ через coincurve (самая быстрая библиотека)
            private_key_bytes = private_key_int.to_bytes(32, 'big')
            privkey = PrivateKey(private_key_bytes)
            
            # Получаем публичный ключ
            pubkey = privkey.public_key
            
            if compressed:
                pubkey_bytes = pubkey.format(compressed=True)  # 33 bytes
            else:
                pubkey_bytes = pubkey.format(compressed=False)  # 65 bytes
            
            # Хешируем: SHA256 -> RIPEMD160
            sha256_hash = hashlib.sha256(pubkey_bytes).digest()
            ripemd160 = hashlib.new('ripemd160')
            ripemd160.update(sha256_hash)
            pubkey_hash = ripemd160.digest()  # 20 bytes
            
            # P2PKH адрес (начинается с '1')
            if address_type == 'p2pkh':
                version_byte = b'\x00'  # Mainnet P2PKH
                versioned_hash = version_byte + pubkey_hash
                
                # Контрольная сумма (double SHA256)
                checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
                
                # Финальный адрес
                full_hash = versioned_hash + checksum
                return base58.b58encode(full_hash).decode()
            
        except Exception:
            pass
        
        return None
    
    def private_key_to_wif(self, private_key_int, compressed=True):
        """Приватный ключ в WIF формат"""
        try:
            # Mainnet prefix
            extended = b'\x80' + private_key_int.to_bytes(32, 'big')
            
            # Добавляем флаг сжатия
            if compressed:
                extended += b'\x01'
            
            # Контрольная сумма
            checksum = hashlib.sha256(hashlib.sha256(extended).digest()).digest()[:4]
            
            return base58.b58encode(extended + checksum).decode()
            
        except Exception:
            pass
        
        return None
    
    def check_balance_api(self, address):
        """Проверка баланса через современные API"""
        apis = [
            f"https://blockstream.info/api/address/{address}",
            f"https://api.blockcypher.com/v1/btc/main/addrs/{address}",
        ]
        
        for api_url in apis:
            try:
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Blockstream API
                    if 'blockstream.info' in api_url:
                        funded = data.get('chain_stats', {}).get('funded_txo_sum', 0)
                        return funded / 100000000  # Satoshi to BTC
                    
                    # BlockCypher API
                    elif 'blockcypher.com' in api_url:
                        balance = data.get('balance', 0)
                        return balance / 100000000  # Satoshi to BTC
                        
            except Exception:
                continue
        
        return 0
    
    def ultimate_worker(self, worker_id, puzzle_num, chunk_start, chunk_end):
        """Ультимативный воркер с максимальной производительностью"""
        puzzle_info = self.bitcoin_puzzles[puzzle_num]
        target_address = puzzle_info['address']
        
        print(f"🚀 Worker {worker_id} attacking Puzzle {puzzle_num}")
        print(f"   Target: {target_address}")
        print(f"   Range: {hex(chunk_start)} - {hex(chunk_end)}")
        
        local_tested = 0
        local_generated = 0
        
        current_key = chunk_start
        
        while self.running.value and current_key <= chunk_end:
            try:
                # Тестируем сжатый адрес
                address_compressed = self.private_key_to_address(current_key, compressed=True)
                local_generated += 1
                
                if address_compressed == target_address:
                    print(f"\n🎉🎉🎉 PUZZLE {puzzle_num} SOLVED! 🎉🎉🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private Key: {hex(current_key)}")
                    print(f"WIF: {self.private_key_to_wif(current_key, True)}")
                    print(f"Address: {address_compressed} (COMPRESSED)")
                    
                    # Проверяем баланс
                    balance = self.check_balance_api(address_compressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем результат
                    result = {
                        'puzzle': puzzle_num,
                        'worker_id': worker_id,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.private_key_to_wif(current_key, True),
                        'address': address_compressed,
                        'compressed': True,
                        'balance_btc': balance,
                        'reward': puzzle_info['reward'],
                        'found_timestamp': time.time(),
                        'found_datetime': time.ctime()
                    }
                    
                    # Сохраняем в файл
                    filename = f"ultimate_results/PUZZLE_{puzzle_num}_SOLVED_{worker_id}.json"
                    with open(filename, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.results.append(result)
                    self.running.value = 0  # Останавливаем все воркеры
                    return True
                
                # Тестируем несжатый адрес
                address_uncompressed = self.private_key_to_address(current_key, compressed=False)
                local_generated += 1
                
                if address_uncompressed == target_address:
                    print(f"\n🎉🎉🎉 PUZZLE {puzzle_num} SOLVED! 🎉🎉🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private Key: {hex(current_key)}")
                    print(f"WIF: {self.private_key_to_wif(current_key, False)}")
                    print(f"Address: {address_uncompressed} (UNCOMPRESSED)")
                    
                    # Проверяем баланс
                    balance = self.check_balance_api(address_uncompressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем результат
                    result = {
                        'puzzle': puzzle_num,
                        'worker_id': worker_id,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.private_key_to_wif(current_key, False),
                        'address': address_uncompressed,
                        'compressed': False,
                        'balance_btc': balance,
                        'reward': puzzle_info['reward'],
                        'found_timestamp': time.time(),
                        'found_datetime': time.ctime()
                    }
                    
                    # Сохраняем в файл
                    filename = f"ultimate_results/PUZZLE_{puzzle_num}_SOLVED_{worker_id}.json"
                    with open(filename, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.results.append(result)
                    self.running.value = 0  # Останавливаем все воркеры
                    return True
                
            except Exception:
                pass
            
            current_key += 1
            local_tested += 2  # Протестировали 2 адреса (сжатый + несжатый)
            
            # Обновляем глобальную статистику каждые 25000 ключей
            if local_tested % 50000 == 0:
                self.stats['keys_tested'] += 50000
                self.stats['addresses_generated'] += local_generated
                local_generated = 0
                
                # Отчет каждые 60 секунд
                current_time = time.time()
                if current_time - self.stats['last_report_time'] > 60:
                    elapsed = current_time - self.stats['start_time']
                    keys_rate = self.stats['keys_tested'] / elapsed
                    addr_rate = self.stats['addresses_generated'] / elapsed
                    
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    print(f"⚡ Worker {worker_id}: {local_tested:,} keys tested")
                    print(f"📊 Global: {keys_rate:.0f} keys/s, {addr_rate:.0f} addr/s")
                    print(f"💻 CPU: {cpu_percent}%, RAM: {memory_percent}%")
                    
                    self.stats['last_report_time'] = current_time
        
        return False
    
    def attack_puzzle(self, puzzle_num=66, max_workers=None):
        """Атака на конкретный Bitcoin Puzzle"""
        if puzzle_num not in self.bitcoin_puzzles:
            print(f"❌ Puzzle {puzzle_num} not available")
            return False
        
        if max_workers is None:
            max_workers = min(self.cpu_count, 32)  # Больше потоков для I/O
        
        puzzle_info = self.bitcoin_puzzles[puzzle_num]
        
        print(f"\n🎯 ATTACKING BITCOIN PUZZLE {puzzle_num}")
        print(f"🎁 Reward: {puzzle_info['reward']}")
        print(f"📍 Target: {puzzle_info['address']}")
        print(f"👥 Workers: {max_workers}")
        print(f"📊 Key space: {hex(puzzle_info['range_start'])} - {hex(puzzle_info['range_end'])}")
        print("-" * 60)
        
        # Разбиваем диапазон на части
        total_range = puzzle_info['range_end'] - puzzle_info['range_start']
        chunk_size = total_range // max_workers
        
        # Запускаем воркеры
        processes = []
        
        for i in range(max_workers):
            chunk_start = puzzle_info['range_start'] + i * chunk_size
            chunk_end = puzzle_info['range_start'] + (i + 1) * chunk_size
            if i == max_workers - 1:  # Последний воркер берет остаток
                chunk_end = puzzle_info['range_end']
            
            process = mp.Process(
                target=self.ultimate_worker,
                args=(i, puzzle_num, chunk_start, chunk_end)
            )
            processes.append(process)
            process.start()
        
        try:
            # Ждем результата
            for process in processes:
                process.join()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping attack...")
            self.running.value = 0
            
            # Останавливаем все процессы
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        
        # Финальная статистика
        elapsed = time.time() - self.stats['start_time']
        print(f"\n📊 FINAL STATISTICS")
        print(f"⏱️  Time: {elapsed:.1f} seconds")
        print(f"🔢 Keys tested: {self.stats['keys_tested']:,}")
        print(f"📍 Addresses generated: {self.stats['addresses_generated']:,}")
        print(f"⚡ Average speed: {self.stats['keys_tested']/elapsed:.0f} keys/second")
        
        if self.results:
            print(f"🎉 PUZZLE SOLVED! Results: {len(self.results)}")
            return True
        else:
            print("❌ Puzzle not solved in this session")
            return False

def main():
    print("🏆 ULTIMATE Bitcoin Puzzle Solver")
    print("Using the most modern crypto libraries and validation methods")
    print("=" * 60)
    
    solver = UltimateBitcoinSolver()
    
    # Выбираем puzzle (66 - разумная цель)
    target_puzzle = 66
    print(f"\nTargeting Puzzle {target_puzzle} with maximum performance...")
    
    success = solver.attack_puzzle(puzzle_num=target_puzzle)
    
    if success:
        print("\n🎉 SUCCESS! Bitcoin Puzzle solved!")
        print("Check ultimate_results/ directory for details")
    else:
        print("\n⏹️  Attack session completed")

if __name__ == "__main__":
    main()
