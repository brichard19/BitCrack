#!/usr/bin/env python3
"""
Продвинутый API клиент для проверки балансов
Поддерживает множественные провайдеры с failover
"""

import requests
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from queue import Queue, Empty

@dataclass
class BalanceResult:
    """Результат проверки баланса"""
    address: str
    balance: float
    network: str
    provider: str
    success: bool
    error: Optional[str] = None
    timestamp: float = 0.0

class APIRateLimiter:
    """Rate limiter для API запросов"""
    
    def __init__(self, max_requests: int = 5, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Получение разрешения на запрос"""
        with self.lock:
            now = time.time()
            
            # Удаляем старые запросы
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Проверяем лимит
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def wait_if_needed(self):
        """Ожидание если превышен лимит"""
        while not self.acquire():
            time.sleep(0.1)

class BitcoinAPIProvider:
    """Провайдер Bitcoin API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limiter = APIRateLimiter(max_requests=10, time_window=1.0)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AdvancedPuzzleSolver/2.0'
        })
    
    def check_balance_blockstream(self, address: str) -> BalanceResult:
        """Проверка через blockstream.info"""
        try:
            self.rate_limiter.wait_if_needed()
            
            url = f"https://blockstream.info/api/address/{address}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Получаем баланс из funded_txo_sum
                chain_stats = data.get('chain_stats', {})
                balance_satoshi = chain_stats.get('funded_txo_sum', 0) - chain_stats.get('spent_txo_sum', 0)
                balance_btc = balance_satoshi / 100000000
                
                return BalanceResult(
                    address=address,
                    balance=balance_btc,
                    network='Bitcoin',
                    provider='blockstream.info',
                    success=True,
                    timestamp=time.time()
                )
                
        except Exception as e:
            logging.error(f"Blockstream API error for {address}: {e}")
        
        return BalanceResult(
            address=address,
            balance=0.0,
            network='Bitcoin',
            provider='blockstream.info',
            success=False,
            error="API request failed",
            timestamp=time.time()
        )
    
    def check_balance_blockcypher(self, address: str) -> BalanceResult:
        """Проверка через blockcypher.com"""
        try:
            self.rate_limiter.wait_if_needed()
            
            url = f"https://api.blockcypher.com/v1/btc/main/addrs/{address}/balance"
            if self.api_key:
                url += f"?token={self.api_key}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                balance_satoshi = data.get('balance', 0)
                balance_btc = balance_satoshi / 100000000
                
                return BalanceResult(
                    address=address,
                    balance=balance_btc,
                    network='Bitcoin',
                    provider='blockcypher.com',
                    success=True,
                    timestamp=time.time()
                )
                
        except Exception as e:
            logging.error(f"BlockCypher API error for {address}: {e}")
        
        return BalanceResult(
            address=address,
            balance=0.0,
            network='Bitcoin',
            provider='blockcypher.com',
            success=False,
            error="API request failed",
            timestamp=time.time()
        )
    
    def check_balance(self, address: str) -> BalanceResult:
        """Проверка баланса с failover"""
        # Пробуем blockstream.info первым
        result = self.check_balance_blockstream(address)
        if result.success:
            return result
        
        # Fallback на blockcypher
        result = self.check_balance_blockcypher(address)
        return result

class EthereumAPIProvider:
    """Провайдер Ethereum API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limiter = APIRateLimiter(max_requests=5, time_window=1.0)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AdvancedPuzzleSolver/2.0'
        })
    
    def check_balance_etherscan(self, address: str) -> BalanceResult:
        """Проверка через etherscan.io"""
        try:
            self.rate_limiter.wait_if_needed()
            
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'account',
                'action': 'balance',
                'address': address,
                'tag': 'latest'
            }
            
            if self.api_key:
                params['apikey'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == '1':
                    balance_wei = int(data.get('result', '0'))
                    balance_eth = balance_wei / 10**18
                    
                    return BalanceResult(
                        address=address,
                        balance=balance_eth,
                        network='Ethereum',
                        provider='etherscan.io',
                        success=True,
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            logging.error(f"Etherscan API error for {address}: {e}")
        
        return BalanceResult(
            address=address,
            balance=0.0,
            network='Ethereum',
            provider='etherscan.io',
            success=False,
            error="API request failed",
            timestamp=time.time()
        )
    
    def check_balance(self, address: str) -> BalanceResult:
        """Проверка баланса Ethereum"""
        return self.check_balance_etherscan(address)

class AdvancedBalanceChecker:
    """Продвинутый проверщик балансов с кешированием"""
    
    def __init__(self, bitcoin_api_key: Optional[str] = None, ethereum_api_key: Optional[str] = None):
        self.bitcoin_provider = BitcoinAPIProvider(bitcoin_api_key)
        self.ethereum_provider = EthereumAPIProvider(ethereum_api_key)
        
        # Кеш результатов
        self.balance_cache: Dict[str, BalanceResult] = {}
        self.cache_duration = 300  # 5 минут
        self.cache_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        logging.info("💰 Продвинутый проверщик балансов инициализирован")
    
    def _is_cache_valid(self, result: BalanceResult) -> bool:
        """Проверка актуальности кеша"""
        return time.time() - result.timestamp < self.cache_duration
    
    def _get_from_cache(self, address: str) -> Optional[BalanceResult]:
        """Получение из кеша"""
        with self.cache_lock:
            if address in self.balance_cache:
                result = self.balance_cache[address]
                if self._is_cache_valid(result):
                    self.stats['cache_hits'] += 1
                    return result
                else:
                    del self.balance_cache[address]
        return None
    
    def _save_to_cache(self, result: BalanceResult):
        """Сохранение в кеш"""
        with self.cache_lock:
            self.balance_cache[result.address] = result
    
    def check_balance(self, address: str, network: str) -> BalanceResult:
        """Проверка баланса с кешированием"""
        # Проверяем кеш
        cached_result = self._get_from_cache(address)
        if cached_result:
            return cached_result
        
        self.stats['total_requests'] += 1
        
        # Определяем провайдера
        if network.lower() == 'bitcoin':
            result = self.bitcoin_provider.check_balance(address)
        elif network.lower() == 'ethereum':
            result = self.ethereum_provider.check_balance(address)
        else:
            result = BalanceResult(
                address=address,
                balance=0.0,
                network=network,
                provider='unknown',
                success=False,
                error=f"Unsupported network: {network}",
                timestamp=time.time()
            )
        
        # Обновляем статистику
        if result.success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['errors'] += 1
        
        # Сохраняем в кеш
        if result.success:
            self._save_to_cache(result)
        
        return result
    
    def check_multiple_balances(self, addresses: List[Tuple[str, str]], max_workers: int = 5) -> List[BalanceResult]:
        """Проверка множественных балансов параллельно"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи
            future_to_addr = {
                executor.submit(self.check_balance, addr, network): (addr, network)
                for addr, network in addresses
            }
            
            # Собираем результаты
            for future in as_completed(future_to_addr):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    addr, network = future_to_addr[future]
                    logging.error(f"Error checking balance for {addr}: {e}")
                    results.append(BalanceResult(
                        address=addr,
                        balance=0.0,
                        network=network,
                        provider='error',
                        success=False,
                        error=str(e),
                        timestamp=time.time()
                    ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики"""
        stats = self.stats.copy()
        stats['cache_size'] = len(self.balance_cache)
        stats['success_rate'] = (
            stats['successful_requests'] / max(stats['total_requests'], 1) * 100
        )
        return stats
    
    def clear_cache(self):
        """Очистка кеша"""
        with self.cache_lock:
            self.balance_cache.clear()
        logging.info("🗑️ Кеш балансов очищен")

# Глобальный проверщик
_balance_checker = None

def get_balance_checker(bitcoin_api_key: Optional[str] = None, ethereum_api_key: Optional[str] = None) -> AdvancedBalanceChecker:
    """Получение глобального проверщика балансов"""
    global _balance_checker
    if _balance_checker is None:
        _balance_checker = AdvancedBalanceChecker(bitcoin_api_key, ethereum_api_key)
    return _balance_checker

if __name__ == "__main__":
    # Тестирование API клиента
    import logging
    logging.basicConfig(level=logging.INFO)
    
    checker = AdvancedBalanceChecker()
    
    test_addresses = [
        ("1By8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "bitcoin"),  # Puzzle #71
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "bitcoin"),  # Genesis block
        ("0x742d35Cc6634C0532925a3b8D0C0Dc4A4b9c9E5A", "ethereum")  # Test ETH
    ]
    
    print("💰 Тестирование проверщика балансов:")
    print("=" * 50)
    
    for address, network in test_addresses:
        print(f"\n🔍 Проверка {network.upper()} адреса:")
        print(f"📍 {address}")
        
        result = checker.check_balance(address, network)
        
        if result.success:
            print(f"✅ Баланс: {result.balance} {network.upper()}")
            print(f"🌐 Провайдер: {result.provider}")
        else:
            print(f"❌ Ошибка: {result.error}")
    
    # Статистика
    print(f"\n📊 Статистика:")
    stats = checker.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
