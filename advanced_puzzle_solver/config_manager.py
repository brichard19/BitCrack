#!/usr/bin/env python3
"""
Менеджер конфигурации для Bitcoin Puzzle Solver
Управляет настройками, API ключами и целевыми адресами
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import logging

@dataclass
class APIConfig:
    """Конфигурация API ключей"""
    etherscan_key: Optional[str] = None
    blockcypher_key: Optional[str] = None
    infura_key: Optional[str] = None
    alchemy_key: Optional[str] = None
    
    def load_from_file(self, config_file: str = "api_config.json") -> bool:
        """Загрузка API ключей из файла"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    
                api_keys = data.get('api_keys', {})
                eth_keys = api_keys.get('ethereum', {})
                btc_keys = api_keys.get('bitcoin', {})
                
                self.etherscan_key = eth_keys.get('etherscan')
                self.infura_key = eth_keys.get('infura')
                self.alchemy_key = eth_keys.get('alchemy')
                self.blockcypher_key = btc_keys.get('blockcypher')
                
                return True
        except Exception as e:
            logging.error(f"Ошибка загрузки API конфигурации: {e}")
        
        return False

@dataclass
class PuzzleTarget:
    """Целевой адрес puzzle"""
    name: str
    address: str
    start_range: int
    end_range: int
    reward_btc: float
    private_key: Optional[str] = None
    found: bool = False

class ConfigManager:
    """Менеджер конфигурации решателя"""
    
    def __init__(self, config_dir: str = "."):
        self.config_dir = config_dir
        self.api_config = APIConfig()
        self.puzzle_targets: List[PuzzleTarget] = []
        self.known_addresses: List[str] = []
        self.network_settings: Dict[str, Any] = {}
        
        self.load_all_configs()
    
    def load_all_configs(self):
        """Загрузка всех конфигураций"""
        # API конфигурация
        api_config_path = os.path.join(self.config_dir, "api_config.json")
        self.api_config.load_from_file(api_config_path)
        
        # Основная конфигурация
        main_config_path = os.path.join(self.config_dir, "puzzle_config.json")
        self.load_main_config(main_config_path)
    
    def load_main_config(self, config_file: str):
        """Загрузка основной конфигурации"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                # Загружаем puzzle цели
                puzzle_71 = data.get('puzzle_71', {})
                puzzle_72 = data.get('puzzle_72', {})
                
                self.puzzle_targets = [
                    PuzzleTarget(
                        name="Puzzle #71",
                        address=puzzle_71.get('known_address', ''),
                        start_range=int(puzzle_71.get('range_start', '0x40000000000000000'), 16),
                        end_range=int(puzzle_71.get('range_end', '0x7FFFFFFFFFFFFFFFFF'), 16),
                        reward_btc=puzzle_71.get('reward_btc', 7.1)
                    ),
                    PuzzleTarget(
                        name="Puzzle #72",
                        address=puzzle_72.get('known_address', ''),
                        start_range=int(puzzle_72.get('range_start', '0x80000000000000000'), 16),
                        end_range=int(puzzle_72.get('range_end', '0xFFFFFFFFFFFFFFFFF'), 16),
                        reward_btc=puzzle_72.get('reward_btc', 7.2)
                    )
                ]
                
                # Загружаем известные адреса с балансами
                known_addresses = data.get('known_addresses_with_balance', [])
                self.known_addresses = [addr['address'] for addr in known_addresses]
                
                # Добавляем target адреса
                target_addresses = data.get('target_addresses', [])
                self.known_addresses.extend(target_addresses)
                
                logging.info(f"Загружено {len(self.puzzle_targets)} puzzle целей")
                logging.info(f"Загружено {len(self.known_addresses)} известных адресов")
                
        except Exception as e:
            logging.error(f"Ошибка загрузки основной конфигурации: {e}")
    
    def get_puzzle_by_name(self, name: str) -> Optional[PuzzleTarget]:
        """Получение puzzle по имени"""
        for puzzle in self.puzzle_targets:
            if puzzle.name.lower() == name.lower():
                return puzzle
        return None
    
    def get_active_puzzle(self) -> PuzzleTarget:
        """Получение активного puzzle (по умолчанию #71)"""
        puzzle = self.get_puzzle_by_name("Puzzle #71")
        if puzzle:
            return puzzle
        
        # Резервный вариант
        return PuzzleTarget(
            name="Default Puzzle #71",
            address="1By8GQbnueYofwSuFAT3USAhGjPrkxDdW9",
            start_range=0x40000000000000000,
            end_range=0x7FFFFFFFFFFFFFFFFF,
            reward_btc=7.1
        )
    
    def add_found_address(self, address: str, private_key: str):
        """Добавление найденного адреса"""
        # Проверяем, является ли это puzzle адресом
        for puzzle in self.puzzle_targets:
            if puzzle.address == address:
                puzzle.private_key = private_key
                puzzle.found = True
                logging.critical(f"🎉 НАЙДЕН {puzzle.name}! Приватный ключ: {private_key}")
                
                # Сохраняем в отдельный файл
                self.save_found_puzzle(puzzle)
                return
    
    def save_found_puzzle(self, puzzle: PuzzleTarget):
        """Сохранение найденного puzzle"""
        found_file = os.path.join(self.config_dir, f"FOUND_{puzzle.name.replace(' ', '_').replace('#', '')}.json")
        
        try:
            found_data = {
                'puzzle': puzzle.name,
                'address': puzzle.address,
                'private_key': puzzle.private_key,
                'reward_btc': puzzle.reward_btc,
                'found_timestamp': time.time(),
                'range_start': hex(puzzle.start_range),
                'range_end': hex(puzzle.end_range)
            }
            
            with open(found_file, 'w') as f:
                json.dump(found_data, f, indent=2)
            
            logging.critical(f"💾 НАЙДЕННЫЙ PUZZLE СОХРАНЕН В: {found_file}")
            
        except Exception as e:
            logging.error(f"Ошибка сохранения найденного puzzle: {e}")
    
    def is_target_address(self, address: str) -> bool:
        """Проверка, является ли адрес целевым"""
        return address in self.known_addresses or any(p.address == address for p in self.puzzle_targets)
    
    def get_api_endpoints(self) -> Dict[str, List[str]]:
        """Получение API endpoints с ключами"""
        endpoints = {
            'bitcoin': [
                'https://blockstream.info/api/address/{address}',
                'https://api.blockcypher.com/v1/btc/main/addrs/{address}/balance'
            ],
            'ethereum': [
                'https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest'
            ]
        }
        
        # Добавляем API ключи если есть
        if self.api_config.blockcypher_key:
            endpoints['bitcoin'].append(
                f'https://api.blockcypher.com/v1/btc/main/addrs/{{address}}/balance?token={self.api_config.blockcypher_key}'
            )
        
        if self.api_config.etherscan_key:
            endpoints['ethereum'][0] += f'&apikey={self.api_config.etherscan_key}'
        
        return endpoints
    
    def export_config(self, filename: str = "exported_config.json"):
        """Экспорт полной конфигурации"""
        export_data = {
            'api_config': asdict(self.api_config),
            'puzzle_targets': [asdict(p) for p in self.puzzle_targets],
            'known_addresses': self.known_addresses,
            'export_timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logging.info(f"Конфигурация экспортирована в: {filename}")

# Глобальный менеджер конфигурации
config_manager = None

def get_config_manager(config_dir: str = ".") -> ConfigManager:
    """Получение глобального менеджера конфигурации"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager(config_dir)
    return config_manager

if __name__ == "__main__":
    # Тест менеджера конфигурации
    import logging
    logging.basicConfig(level=logging.INFO)
    
    manager = ConfigManager(".")
    
    print("🔧 Менеджер конфигурации загружен")
    print(f"📊 Puzzle целей: {len(manager.puzzle_targets)}")
    print(f"📍 Известных адресов: {len(manager.known_addresses)}")
    
    active_puzzle = manager.get_active_puzzle()
    print(f"🎯 Активный puzzle: {active_puzzle.name}")
    print(f"💰 Награда: {active_puzzle.reward_btc} BTC")
    print(f"🔍 Диапазон: {hex(active_puzzle.start_range)} - {hex(active_puzzle.end_range)}")
    
    # Тест API endpoints
    endpoints = manager.get_api_endpoints()
    print(f"🌐 Bitcoin API endpoints: {len(endpoints['bitcoin'])}")
    print(f"🌐 Ethereum API endpoints: {len(endpoints['ethereum'])}")
