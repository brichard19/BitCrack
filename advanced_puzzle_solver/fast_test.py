#!/usr/bin/env python3
"""
🔥 FAST TEST SOLVER - Быстрая версия для тестирования 🔥
"""

import hashlib
import threading
import time
import secrets
import logging
import json
import os
import signal
import sys
from typing import Set
import multiprocessing as mp

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastSolver:
    """Быстрый тестовый решатель"""
    
    def __init__(self):
        # Константы secp256k1
        self.P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        self.N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        self.G = (
            0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
            0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        )
        
        # Puzzle адреса для тестирования
        self.targets = {
            "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",  # Puzzle 60
            "1CUhaTe2wecqRuqPiqueXstmK6P4Lkoz3J",  # Puzzle 61
            "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",  # Puzzle 63
        }
        
        self.is_running = False
        self.total_keys = mp.Value('Q', 0)
        self.start_time = time.time()
        
        # Количество потоков
        self.num_threads = min(mp.cpu_count() * 2, 20)
        
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        
        logger.info(f"🔥 Fast Solver готов! Потоков: {self.num_threads}")
    
    def _stop(self, signum, frame):
        """Остановка"""
        logger.info("🛑 Остановка...")
        self.is_running = False
        self._report()
        sys.exit(0)
    
    def _report(self):
        """Отчет"""
        runtime = time.time() - self.start_time
        speed = self.total_keys.value / runtime if runtime > 0 else 0
        
        logger.info(f"""
🏁 === ФИНАЛЬНЫЙ ОТЧЕТ ===
⚡ Скорость: {speed:,.0f} ключей/сек
🔢 Всего: {self.total_keys.value:,} ключей  
⏰ Время: {runtime:.1f} сек
========================
""")
    
    def mod_inverse(self, a, m):
        """Модульное обратное"""
        return pow(a, m - 2, m)
    
    def point_add(self, p1, p2):
        """Сложение точек"""
        if p1 is None: return p2
        if p2 is None: return p1
        
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:
            if y1 == y2:
                s = (3 * x1 * x1 * self.mod_inverse(2 * y1, self.P)) % self.P
            else:
                return None
        else:
            s = ((y2 - y1) * self.mod_inverse(x2 - x1, self.P)) % self.P
        
        x3 = (s * s - x1 - x2) % self.P
        y3 = (s * (x1 - x3) - y1) % self.P
        
        return (x3, y3)
    
    def point_multiply(self, k, point):
        """Умножение точки"""
        if k == 0: return None
        if k == 1: return point
        
        result = None
        addend = point
        
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1
        
        return result
    
    def point_to_address(self, point, compressed=True):
        """Точка в адрес"""
        if point is None: return ""
        
        try:
            x, y = point
            x_bytes = x.to_bytes(32, 'big')
            
            if compressed:
                if y % 2 == 0:
                    pub_key = b'\x02' + x_bytes
                else:
                    pub_key = b'\x03' + x_bytes
            else:
                y_bytes = y.to_bytes(32, 'big')
                pub_key = b'\x04' + x_bytes + y_bytes
            
            # SHA256 + RIPEMD160
            sha = hashlib.sha256(pub_key).digest()
            rmd = hashlib.new('ripemd160', sha).digest()
            
            # Base58Check
            versioned = b'\x00' + rmd
            checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
            full = versioned + checksum
            
            # Base58 encode
            alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            num = int.from_bytes(full, 'big')
            
            encoded = ""
            while num > 0:
                num, rem = divmod(num, 58)
                encoded = alphabet[rem] + encoded
            
            # Добавить ведущие единицы
            for byte in full:
                if byte == 0:
                    encoded = alphabet[0] + encoded
                else:
                    break
            
            return encoded
            
        except Exception:
            return ""
    
    def worker(self, worker_id, start, end):
        """Воркер"""
        logger.info(f"⚡ Воркер {worker_id} запущен: {hex(start)} - {hex(end)}")
        
        local_count = 0
        last_report = time.time()
        
        current = start + worker_id * 1000
        step = self.num_threads * 1000
        
        try:
            while self.is_running and current < end:
                # Батч обработка
                for offset in range(500):
                    if not self.is_running:
                        break
                    
                    key = current + offset
                    if key >= end:
                        break
                    
                    if 1 <= key < self.N:
                        point = self.point_multiply(key, self.G)
                        
                        if point:
                            # Проверяем оба формата
                            for compressed in [True, False]:
                                addr = self.point_to_address(point, compressed)
                                
                                if addr in self.targets:
                                    logger.critical(f"🎯 НАЙДЕН! {addr} -> {hex(key)}")
                                    
                                    # Сохранить результат
                                    result = {
                                        'address': addr,
                                        'private_key': hex(key),
                                        'timestamp': time.time()
                                    }
                                    with open(f'FOUND_{int(time.time())}.json', 'w') as f:
                                        json.dump(result, f, indent=2)
                    
                    local_count += 1
                
                current += step
                
                # Обновить общий счетчик
                with self.total_keys.get_lock():
                    self.total_keys.value += 500
                
                # Отчет каждые 10 секунд
                if time.time() - last_report > 10:
                    runtime = time.time() - self.start_time
                    speed = local_count / runtime if runtime > 0 else 0
                    logger.info(f"🔥 Воркер {worker_id}: {local_count:,} ключей, {speed:.0f} к/сек")
                    last_report = time.time()
                    
        except Exception as e:
            logger.error(f"Ошибка в воркере {worker_id}: {e}")
        
        logger.info(f"✅ Воркер {worker_id} завершен: {local_count:,} ключей")
    
    def start(self):
        """Запуск"""
        logger.info("🚀 БЫСТРЫЙ ЗАПУСК!")
        
        self.is_running = True
        
        # Диапазон puzzle 63
        start = 0x4000000000000000
        end = 0x7FFFFFFFFFFFFFFF
        
        workers = []
        
        # Запуск воркеров
        for i in range(self.num_threads):
            w = threading.Thread(
                target=self.worker,
                args=(i, start, end),
                daemon=True
            )
            workers.append(w)
            w.start()
        
        logger.info(f"🧵 Запущено {len(workers)} воркеров")
        
        # Главный цикл
        try:
            while self.is_running:
                time.sleep(30)  # Отчет каждые 30 сек
                
                runtime = time.time() - self.start_time
                speed = self.total_keys.value / runtime if runtime > 0 else 0
                alive = sum(1 for w in workers if w.is_alive())
                
                logger.info(f"📊 Скорость: {speed:,.0f} к/сек, Ключей: {self.total_keys.value:,}, Потоков: {alive}")
                
        except KeyboardInterrupt:
            logger.info("🛑 Ctrl+C получен")
            self.is_running = False
        
        # Ожидание завершения
        for w in workers:
            if w.is_alive():
                w.join(timeout=2)
        
        self._report()
        logger.info("🏁 ТЕСТ ЗАВЕРШЕН!")

def main():
    print("🔥" * 40)
    print("🔥 FAST TEST SOLVER 🔥")
    print("🔥" * 40)
    
    solver = FastSolver()
    solver.start()

if __name__ == "__main__":
    main()
