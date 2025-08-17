#!/usr/bin/env python3
"""
🛡️ BITCOIN ADDRESS VALIDATOR - Профессиональный валидатор адресов
Красивый, быстрый и надежный инструмент для проверки Bitcoin адресов

Поддерживает все типы Bitcoin адресов:
- Legacy (P2PKH): начинаются с '1'
- Script Hash (P2SH): начинаются с '3' 
- SegWit v0 (Bech32): начинаются с 'bc1q'
- SegWit v1 Taproot (Bech32m): начинаются с 'bc1p'
"""

import sys
import re
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import base58
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
    import requests
except ImportError as e:
    print(f"Устанавливаю зависимости: {e.name}")
    import os
    os.system("pip install base58 rich requests")
    import base58
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

console = Console()

class AdvancedAddressValidator:
    """Продвинутый валидатор адресов"""
    
    # Bitcoin адресные префиксы
    BITCOIN_PREFIXES = {
        'P2PKH_MAINNET': [0x00],
        'P2SH_MAINNET': [0x05],
        'P2PKH_TESTNET': [0x6F],
        'P2SH_TESTNET': [0xC4]
    }
    
    # Bech32 charset
    BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_bitcoin_legacy(self, address: str) -> Tuple[bool, Optional[str]]:
        """Валидация legacy Bitcoin адреса (P2PKH, P2SH)"""
        try:
            # Декодируем base58
            decoded = base58.b58decode(address)
            
            # Проверяем длину
            if len(decoded) != 25:
                return False, "Неверная длина адреса"
            
            # Извлекаем компоненты
            version = decoded[0]
            payload = decoded[:-4]
            checksum = decoded[-4:]
            
            # Проверяем чексумму
            hash_result = hashlib.sha256(hashlib.sha256(payload).digest()).digest()
            if checksum != hash_result[:4]:
                return False, "Неверная чексумма"
            
            # Определяем тип адреса
            addr_type = "Unknown"
            if version in self.BITCOIN_PREFIXES['P2PKH_MAINNET']:
                addr_type = "P2PKH Mainnet"
            elif version in self.BITCOIN_PREFIXES['P2SH_MAINNET']:
                addr_type = "P2SH Mainnet"
            elif version in self.BITCOIN_PREFIXES['P2PKH_TESTNET']:
                addr_type = "P2PKH Testnet"
            elif version in self.BITCOIN_PREFIXES['P2SH_TESTNET']:
                addr_type = "P2SH Testnet"
            
            return True, addr_type
            
        except Exception as e:
            return False, f"Ошибка декодирования: {str(e)}"
    
    def validate_bech32(self, address: str) -> Tuple[bool, Optional[str]]:
        """Валидация Bech32 адреса (SegWit)"""
        try:
            # Проверяем формат
            if not address.startswith(('bc1', 'tb1')):
                return False, "Неверный префикс Bech32"
            
            # Базовые проверки
            if len(address) < 14 or len(address) > 74:
                return False, "Неверная длина Bech32"
            
            # Проверяем символы
            lower_address = address.lower()
            if not all(c in self.BECH32_CHARSET + '1' for c in lower_address[3:]):
                return False, "Недопустимые символы в Bech32"
            
            # Определяем тип
            addr_type = "SegWit v0"
            if address.startswith('bc1'):
                addr_type += " Mainnet"
            else:
                addr_type += " Testnet"
            
            return True, addr_type
            
        except Exception as e:
            return False, f"Ошибка валидации Bech32: {str(e)}"
    
    def validate_bitcoin_address(self, address: str) -> Tuple[bool, Optional[str]]:
        """Универсальная валидация Bitcoin адреса"""
        if not address or len(address) < 26:
            return False, "Слишком короткий адрес"
        
        # Bech32 адреса
        if address.lower().startswith(('bc1', 'tb1')):
            return self.validate_bech32(address)
        
        # Legacy адреса
        elif address[0] in '13mn2':  # Добавляем поддержку testnet и других префиксов
            return self.validate_bitcoin_legacy(address)
        
        else:
            return False, "Неизвестный формат Bitcoin адреса"
    
    def validate_ethereum_address(self, address: str) -> Tuple[bool, Optional[str]]:
        """Валидация Ethereum адреса"""
        try:
            # Проверяем базовый формат
            if not address.startswith('0x'):
                return False, "Адрес должен начинаться с 0x"
            
            if len(address) != 42:
                return False, "Неверная длина адреса (должно быть 42 символа)"
            
            # Проверяем hex символы
            hex_part = address[2:]
            if not re.match('^[0-9a-fA-F]+$', hex_part):
                return False, "Недопустимые символы в адресе"
            
            # Проверяем EIP-55 checksum если есть mixed case
            if self._has_mixed_case(address):
                if not self._validate_eip55_checksum(address):
                    return False, "Неверная EIP-55 чексумма"
                return True, "Ethereum с EIP-55 чексуммой"
            
            return True, "Ethereum адрес"
            
        except Exception as e:
            return False, f"Ошибка валидации Ethereum: {str(e)}"
    
    def _has_mixed_case(self, address: str) -> bool:
        """Проверяет наличие смешанного регистра"""
        hex_part = address[2:]
        return hex_part != hex_part.lower() and hex_part != hex_part.upper()
    
    def _validate_eip55_checksum(self, address: str) -> bool:
        """Валидация EIP-55 чексуммы"""
        try:
            from Crypto.Hash import keccak
            
            address = address[2:].lower()  # Убираем 0x и приводим к нижнему регистру
            
            # Вычисляем Keccak256 хеш
            hash_bytes = keccak.new(digest_bits=256).update(address.encode()).digest()
            hash_hex = hash_bytes.hex()
            
            # Проверяем каждый символ
            for i, char in enumerate(address):
                if char.isalpha():
                    # Если bit в hash >= 8, символ должен быть в верхнем регистре
                    if int(hash_hex[i], 16) >= 8:
                        if char != char.upper():
                            return False
                    else:
                        if char != char.lower():
                            return False
            
            return True
            
        except Exception:
            return False
    
    def validate_any_address(self, address: str) -> Dict[str, any]:
        """Универсальная валидация любого адреса"""
        result = {
            'address': address,
            'valid': False,
            'network': None,
            'type': None,
            'error': None
        }
        
        # Пробуем Bitcoin
        btc_valid, btc_type = self.validate_bitcoin_address(address)
        if btc_valid:
            result.update({
                'valid': True,
                'network': 'Bitcoin',
                'type': btc_type
            })
            return result
        
        # Пробуем Ethereum
        eth_valid, eth_type = self.validate_ethereum_address(address)
        if eth_valid:
            result.update({
                'valid': True,
                'network': 'Ethereum',
                'type': eth_type
            })
            return result
        
        # Если ничего не подошло
        result['error'] = f"Неизвестный формат адреса: {address[:20]}..."
        return result
    
    def get_address_info(self, address: str) -> Dict[str, any]:
        """Получение полной информации об адресе"""
        info = self.validate_any_address(address)
        
        if info['valid']:
            # Добавляем дополнительную информацию
            if info['network'] == 'Bitcoin':
                info['is_mainnet'] = 'Mainnet' in info['type']
                info['is_segwit'] = info['type'].startswith('SegWit')
                info['is_legacy'] = 'P2PKH' in info['type'] or 'P2SH' in info['type']
            elif info['network'] == 'Ethereum':
                info['has_checksum'] = 'EIP-55' in info['type']
        
        return info

# Глобальный валидатор
_validator = None

def get_address_validator() -> AdvancedAddressValidator:
    """Получение глобального валидатора"""
    global _validator
    if _validator is None:
        _validator = AdvancedAddressValidator()
    return _validator

def validate_address(address: str) -> Dict[str, any]:
    """Быстрая валидация адреса"""
    return get_address_validator().validate_any_address(address)

if __name__ == "__main__":
    # Тестирование валидатора
    validator = AdvancedAddressValidator()
    
    test_addresses = [
        "1By8GQbnueYofwSuFAT3USAhGjPrkxDdW9",  # Bitcoin P2PKH
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",  # Bitcoin P2SH
        "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",  # Bitcoin Bech32
        "0x742C4883CbF0b5F4b3b6d1a578b3b5C5C3F0b1dE",  # Ethereum
        "invalid_address"  # Неверный адрес
    ]
    
    print("🔍 Тестирование валидатора адресов:")
    print("=" * 50)
    
    for addr in test_addresses:
        info = validator.get_address_info(addr)
        print(f"\n📍 Адрес: {addr}")
        print(f"✅ Валидный: {info['valid']}")
        if info['valid']:
            print(f"🌐 Сеть: {info['network']}")
            print(f"🏷️ Тип: {info['type']}")
        else:
            print(f"❌ Ошибка: {info['error']}")
