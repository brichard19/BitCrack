#!/usr/bin/env python3
"""
🔥 WORKING Bitcoin Puzzle Solver - На базе проверенного кода
Возвращается к рабочим библиотекам и методам из первых версий
"""

import os
import sys
import time
import hashlib
import secrets
import threading
import json
from concurrent.futures import ThreadPoolExecutor
import requests

# Используем только ПРОВЕРЕННЫЕ библиотеки из первых версий
try:
    import ecdsa
    import base58
    from Crypto.Hash import RIPEMD160
    print("✅ Using proven libraries: ecdsa, base58, pycryptodome")
except ImportError as e:
    print(f"Installing: {e.name}")
    os.system("pip install ecdsa base58 pycryptodome requests")
    import ecdsa
    import base58
    from Crypto.Hash import RIPEMD160

class WorkingBitcoinSolver:
    """Проверенный рабочий Bitcoin Puzzle Solver"""
    
    def __init__(self):
        self.cpu_count = os.cpu_count()
        
        # НАСТОЯЩИЕ Bitcoin Puzzle адреса
        self.puzzles = {
            64: "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",
            65: "18ZMbwUFLMHoZBbfpCjUJQTCu352eqkeLT",
            66: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
            67: "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9",
        }
        
        self.running = True
        self.total_keys = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        print("🔥 WORKING Bitcoin Puzzle Solver")
        print("=" * 50)
        print(f"💻 CPU cores: {self.cpu_count}")
        print(f"🎯 Puzzles: {list(self.puzzles.keys())}")
        print("🔧 Using PROVEN libraries from working versions")
    
    def private_to_wif(self, private_key_int, compressed=True):
        """WIF формат (как в оригинальных версиях)"""
        # Mainnet prefix
        extended_key = b'\x80' + private_key_int.to_bytes(32, 'big')
        
        if compressed:
            extended_key += b'\x01'
        
        # Double SHA256
        checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
        
        return base58.b58encode(extended_key + checksum).decode()
    
    def private_to_address(self, private_key_int, compressed=True):
        """Проверенное преобразование в адрес (точно как в первых версиях)"""
        try:
            # Создаем приватный ключ через ecdsa (ПРОВЕРЕННАЯ библиотека)
            private_key_bytes = private_key_int.to_bytes(32, 'big')
            signing_key = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
            verifying_key = signing_key.get_verifying_key()
            
            # Публичный ключ
            if compressed:
                x = verifying_key.pubkey.point.x()
                y = verifying_key.pubkey.point.y()
                
                if y % 2 == 0:
                    public_key_bytes = b'\x02' + x.to_bytes(32, 'big')
                else:
                    public_key_bytes = b'\x03' + x.to_bytes(32, 'big')
            else:
                public_key_bytes = b'\x04' + verifying_key.to_string()
            
            # SHA256 -> RIPEMD160
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            ripemd160_hash = RIPEMD160.new(sha256_hash).digest()
            
            # Версия + хеш
            versioned_hash = b'\x00' + ripemd160_hash
            
            # Контрольная сумма
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            # Финальный адрес
            address_bytes = versioned_hash + checksum
            return base58.b58encode(address_bytes).decode()
            
        except Exception:
            return None
    
    def check_balance(self, address):
        """Быстрая проверка баланса"""
        try:
            url = f"https://blockstream.info/api/address/{address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                funded = data.get('chain_stats', {}).get('funded_txo_sum', 0)
                return funded / 100000000  # Satoshi to BTC
                
        except Exception:
            pass
        
        return 0
    
    def working_worker(self, worker_id, puzzle_num, start_range, end_range):
        """РАБОЧИЙ воркер на базе проверенного кода"""
        target_address = self.puzzles[puzzle_num]
        local_count = 0
        
        print(f"🔥 Worker {worker_id} started on Puzzle {puzzle_num}")
        print(f"   Target: {target_address}")
        
        current_key = start_range + worker_id  # Смещение
        
        while self.running and current_key < end_range:
            try:
                # Сжатый адрес
                addr_compressed = self.private_to_address(current_key, compressed=True)
                
                if addr_compressed == target_address:
                    print(f"\n🎉 FOUND PUZZLE {puzzle_num}! 🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private key: {hex(current_key)}")
                    print(f"WIF: {self.private_to_wif(current_key, True)}")
                    print(f"Address: {addr_compressed} (compressed)")
                    
                    # Проверяем баланс
                    balance = self.check_balance(addr_compressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем
                    result = {
                        'puzzle': puzzle_num,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.private_to_wif(current_key, True),
                        'address': addr_compressed,
                        'compressed': True,
                        'balance': balance,
                        'worker': worker_id,
                        'found_at': time.ctime()
                    }
                    
                    with open(f"WORKING_PUZZLE_{puzzle_num}_FOUND.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.running = False
                    return True
                
                # Несжатый адрес
                addr_uncompressed = self.private_to_address(current_key, compressed=False)
                
                if addr_uncompressed == target_address:
                    print(f"\n🎉 FOUND PUZZLE {puzzle_num}! 🎉")
                    print(f"Worker: {worker_id}")
                    print(f"Private key: {hex(current_key)}")
                    print(f"WIF: {self.private_to_wif(current_key, False)}")
                    print(f"Address: {addr_uncompressed} (uncompressed)")
                    
                    # Проверяем баланс
                    balance = self.check_balance(addr_uncompressed)
                    print(f"Balance: {balance} BTC")
                    
                    # Сохраняем
                    result = {
                        'puzzle': puzzle_num,
                        'private_key_hex': hex(current_key),
                        'private_key_wif': self.private_to_wif(current_key, False),
                        'address': addr_uncompressed,
                        'compressed': False,
                        'balance': balance,
                        'worker': worker_id,
                        'found_at': time.ctime()
                    }
                    
                    with open(f"WORKING_PUZZLE_{puzzle_num}_FOUND.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    self.running = False
                    return True
                    
            except Exception:
                pass
            
            current_key += self.cpu_count  # Прыжок на количество воркеров
            local_count += 1
            
            # Статистика каждые 25000
            if local_count % 25000 == 0:
                with self.lock:
                    self.total_keys += 25000
                
                elapsed = time.time() - self.start_time
                rate = local_count / elapsed if elapsed > 0 else 0
                
                print(f"⚡ Worker {worker_id}: {local_count:,} keys, {rate:.0f} k/s")
        
        return False
    
    def attack_puzzle(self, puzzle_num=66, max_workers=None):
        """Атакуем выбранный puzzle"""
        if puzzle_num not in self.puzzles:
            print(f"❌ Puzzle {puzzle_num} not available")
            return False
        
        if max_workers is None:
            max_workers = min(self.cpu_count, 20)
        
        print(f"\n🎯 ATTACKING PUZZLE {puzzle_num}")
        print(f"📍 Target: {self.puzzles[puzzle_num]}")
        print(f"👥 Workers: {max_workers}")
        
        # Определяем диапазон
        if puzzle_num == 64:
            start = 0x8000000000000000
            end = 0xFFFFFFFFFFFFFFFF
        elif puzzle_num == 65:
            start = 0x10000000000000000
            end = 0x1FFFFFFFFFFFFFFFF
        elif puzzle_num == 66:
            start = 0x20000000000000000
            end = 0x3FFFFFFFFFFFFFFFF
        elif puzzle_num == 67:
            start = 0x40000000000000000
            end = 0x7FFFFFFFFFFFFFFFF
        else:
            start = 2**(puzzle_num-1)
            end = 2**puzzle_num - 1
        
        print(f"📊 Range: {hex(start)} - {hex(end)}")
        print("-" * 50)
        
        # Запускаем воркеров
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = (end - start) // max_workers
            
            futures = []
            for i in range(max_workers):
                worker_start = start + i * chunk_size
                worker_end = start + (i + 1) * chunk_size if i < max_workers - 1 else end
                
                future = executor.submit(self.working_worker, i, puzzle_num, worker_start, worker_end)
                futures.append(future)
            
            # Мониторинг
            try:
                while any(not f.done() for f in futures):
                    # Проверяем результаты
                    for f in futures:
                        if f.done() and f.result():
                            print("🎉 PUZZLE SOLVED!")
                            return True
                    
                    time.sleep(10)
                    
                    # Статистика каждые 2 минуты
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        rate = self.total_keys / elapsed
                        active_workers = sum(1 for f in futures if not f.done())
                        print(f"📈 Total: {self.total_keys:,} keys, {rate:.0f} k/s, {active_workers} active")
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                self.running = False
        
        return False

def main():
    print("🔥 WORKING Bitcoin Puzzle Solver")
    print("Based on proven code from successful versions")
    print("=" * 50)
    
    solver = WorkingBitcoinSolver()
    
    # Атакуем puzzle 66 (разумная цель)
    puzzle_target = 66
    
    print(f"\nStarting attack on Puzzle {puzzle_target}...")
    print("This will generate REAL Bitcoin addresses and check balances!")
    
    success = solver.attack_puzzle(puzzle_num=puzzle_target)
    
    if success:
        print("\n🎉 SUCCESS! Puzzle solved!")
    else:
        print("\n⏹️  Attack completed")
    
    elapsed = time.time() - solver.start_time
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Total keys: {solver.total_keys:,}")

if __name__ == "__main__":
    main()
