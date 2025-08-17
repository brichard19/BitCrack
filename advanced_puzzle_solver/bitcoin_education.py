#!/usr/bin/env python3
"""
🎓 ОБРАЗОВАТЕЛЬНЫЙ Bitcoin/Crypto Toolkit
Изучаем криптографию, блокчейн и безопасность на практике

НЕ для поиска puzzle ключей (это невозможно),
А для понимания как работает Bitcoin криптография
"""

import os
import sys
import hashlib
import secrets
import json
import time
from typing import Tuple, Optional

try:
    import ecdsa
    import base58
    from Crypto.Hash import RIPEMD160
    import requests
except ImportError:
    print("Устанавливаю библиотеки...")
    os.system("pip install ecdsa base58 pycryptodome requests")

class BitcoinCryptoEducation:
    """Образовательный класс для изучения Bitcoin криптографии"""
    
    def __init__(self):
        print("🎓 Bitcoin Crypto Education Toolkit")
        print("=" * 50)
        print("Изучаем как работает Bitcoin на практике")
        print("(Не для реального майнинга - это невозможно)")
        print()
    
    def lesson_1_key_generation(self):
        """Урок 1: Как генерируются Bitcoin ключи"""
        print("📚 УРОК 1: Генерация Bitcoin ключей")
        print("-" * 40)
        
        # 1. Генерируем случайный приватный ключ
        private_key_int = secrets.randbelow(2**256)
        private_key_hex = hex(private_key_int)[2:].zfill(64)
        
        print(f"1️⃣ Приватный ключ (256 бит):")
        print(f"   Hex: {private_key_hex}")
        print(f"   Decimal: {private_key_int}")
        
        # 2. Создаем публичный ключ через ECDSA
        sk = ecdsa.SigningKey.from_string(
            bytes.fromhex(private_key_hex), 
            curve=ecdsa.SECP256k1
        )
        vk = sk.get_verifying_key()
        
        # Сжатый формат
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        
        if y % 2 == 0:
            compressed_pubkey = '02' + hex(x)[2:].zfill(64)
        else:
            compressed_pubkey = '03' + hex(x)[2:].zfill(64)
        
        print(f"2️⃣ Публичный ключ (сжатый):")
        print(f"   {compressed_pubkey}")
        
        return private_key_hex, compressed_pubkey
    
    def lesson_2_address_creation(self, public_key_hex: str):
        """Урок 2: Как создается Bitcoin адрес"""
        print("\n📚 УРОК 2: Создание Bitcoin адреса")
        print("-" * 40)
        
        # 1. SHA256 от публичного ключа
        pubkey_bytes = bytes.fromhex(public_key_hex)
        sha256_hash = hashlib.sha256(pubkey_bytes).digest()
        print(f"1️⃣ SHA256(pubkey): {sha256_hash.hex()}")
        
        # 2. RIPEMD160
        ripemd160_hash = RIPEMD160.new(sha256_hash).digest()
        print(f"2️⃣ RIPEMD160: {ripemd160_hash.hex()}")
        
        # 3. Добавляем версию (0x00 для mainnet)
        versioned = b'\x00' + ripemd160_hash
        print(f"3️⃣ + Version byte: {versioned.hex()}")
        
        # 4. Контрольная сумма (double SHA256)
        checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
        print(f"4️⃣ Checksum: {checksum.hex()}")
        
        # 5. Финальный адрес
        final_bytes = versioned + checksum
        address = base58.b58encode(final_bytes).decode()
        print(f"5️⃣ Bitcoin адрес: {address}")
        
        return address
    
    def lesson_3_wif_format(self, private_key_hex: str):
        """Урок 3: WIF формат приватного ключа"""
        print("\n📚 УРОК 3: WIF (Wallet Import Format)")
        print("-" * 40)
        
        # 1. Добавляем префикс mainnet
        extended = b'\x80' + bytes.fromhex(private_key_hex) + b'\x01'  # +01 для compressed
        print(f"1️⃣ Extended key: {extended.hex()}")
        
        # 2. Контрольная сумма
        checksum = hashlib.sha256(hashlib.sha256(extended).digest()).digest()[:4]
        print(f"2️⃣ Checksum: {checksum.hex()}")
        
        # 3. WIF
        wif = base58.b58encode(extended + checksum).decode()
        print(f"3️⃣ WIF: {wif}")
        
        return wif
    
    def lesson_4_address_validation(self, address: str):
        """Урок 4: Валидация Bitcoin адреса"""
        print(f"\n📚 УРОК 4: Валидация адреса {address}")
        print("-" * 40)
        
        try:
            # Декодируем
            decoded = base58.b58decode(address)
            print(f"1️⃣ Decoded bytes: {decoded.hex()}")
            
            # Разделяем на части
            version_hash = decoded[:-4]
            checksum = decoded[-4:]
            print(f"2️⃣ Version+Hash: {version_hash.hex()}")
            print(f"3️⃣ Checksum: {checksum.hex()}")
            
            # Проверяем контрольную сумму
            calculated_checksum = hashlib.sha256(hashlib.sha256(version_hash).digest()).digest()[:4]
            print(f"4️⃣ Calculated checksum: {calculated_checksum.hex()}")
            
            is_valid = checksum == calculated_checksum
            print(f"5️⃣ Валидность: {'✅ VALID' if is_valid else '❌ INVALID'}")
            
            return is_valid
            
        except Exception as e:
            print(f"❌ Ошибка валидации: {e}")
            return False
    
    def lesson_5_security_analysis(self):
        """Урок 5: Анализ безопасности"""
        print("\n📚 УРОК 5: Безопасность Bitcoin")
        print("-" * 40)
        
        print("🔢 Математика безопасности:")
        print(f"   • Размер приватного ключа: 256 бит")
        print(f"   • Возможных ключей: 2^256 = {2**256:,}")
        print(f"   • Это примерно: {2**256 // 10**77:.0e} * 10^77")
        print(f"   • Атомов во вселенной: ~10^80")
        
        print("\n⏰ Временные оценки для брутфорса:")
        speeds = [
            (1000, "Домашний ПК"),
            (1000000, "Мощная система"), 
            (1000000000, "Суперкомпьютер"),
            (10**18, "Все компьютеры мира")
        ]
        
        total_keys = 2**256
        
        for speed, description in speeds:
            years = (total_keys / 2) / speed / (365 * 24 * 3600)
            print(f"   • {description} ({speed:,} к/с): {years:.0e} лет")
        
        print(f"\n🌌 Для сравнения:")
        print(f"   • Возраст вселенной: ~1.4 * 10^10 лет")
        print(f"   • Время до смерти солнца: ~5 * 10^9 лет")
        
        print(f"\n✅ Вывод: Bitcoin криптография НЕВЕРОЯТНО безопасна!")
    
    def lesson_6_real_applications(self):
        """Урок 6: Реальные применения знаний"""
        print("\n📚 УРОК 6: Практические применения")
        print("-" * 40)
        
        applications = [
            "🔐 Создание безопасных кошельков",
            "🏦 Разработка DeFi приложений", 
            "🎨 NFT платформы",
            "🔗 Блокчейн интеграции",
            "🛡️ Криптографические аудиты",
            "📊 Анализ блокчейн данных",
            "🎓 Обучение криптографии",
            "🔧 Инструменты для разработчиков"
        ]
        
        print("Что можно делать с этими знаниями:")
        for app in applications:
            print(f"   {app}")
        
        print(f"\n🎯 Рекомендуемые направления изучения:")
        print(f"   • Solidity (смарт-контракты Ethereum)")
        print(f"   • Web3.py/Web3.js (взаимодействие с блокчейном)")
        print(f"   • Криптографические библиотеки")
        print(f"   • Протоколы консенсуса")
        print(f"   • Безопасность смарт-контракты")
    
    def practical_demo(self):
        """Практическая демонстрация"""
        print("\n🚀 ПРАКТИЧЕСКАЯ ДЕМОНСТРАЦИЯ")
        print("=" * 50)
        
        # Генерируем ключ и показываем весь процесс
        private_key, public_key = self.lesson_1_key_generation()
        address = self.lesson_2_address_creation(public_key)
        wif = self.lesson_3_wif_format(private_key)
        self.lesson_4_address_validation(address)
        self.lesson_5_security_analysis()
        self.lesson_6_real_applications()
        
        print(f"\n📋 ИТОГОВАЯ ИНФОРМАЦИЯ:")
        print(f"Private Key: {private_key}")
        print(f"Public Key:  {public_key}")
        print(f"Address:     {address}")
        print(f"WIF:         {wif}")
        
        # Сохраняем для изучения
        demo_data = {
            'private_key': private_key,
            'public_key': public_key,
            'address': address,
            'wif': wif,
            'timestamp': time.time(),
            'note': 'Сгенерировано для образовательных целей'
        }
        
        with open('bitcoin_crypto_demo.json', 'w') as f:
            json.dump(demo_data, f, indent=2)
        
        print(f"\n💾 Данные сохранены в: bitcoin_crypto_demo.json")

def main():
    """Главная функция"""
    print("🎓 ИЗУЧАЕМ BITCOIN КРИПТОГРАФИЮ")
    print("Честно, без ложных обещаний")
    print()
    
    educator = BitcoinCryptoEducation()
    educator.practical_demo()
    
    print(f"\n" + "="*60)
    print(f"🎯 ЭТО ОБРАЗОВАНИЕ, НЕ МАЙНИНГ!")
    print(f"Puzzle ключи найти невозможно - это математический факт.")
    print(f"Но теперь вы понимаете как работает Bitcoin!")
    print(f"Используйте эти знания для реальных проектов!")
    print(f"="*60)

if __name__ == "__main__":
    main()
