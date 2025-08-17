#!/usr/bin/env python3
"""
🛡️ PROFESSIONAL BITCOIN ADDRESS VALIDATOR
Красивый, полнофункциональный валидатор Bitcoin адресов

Автор: Для пользователя, который заслуживает качественный инструмент
"""

import sys
import hashlib
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import base58
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    print("✅ Все библиотеки загружены успешно!")
except ImportError as e:
    print(f"Устанавливаю зависимость: {e.name}")
    import os
    os.system("pip install base58 rich")
    import base58
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

@dataclass
class ValidationResult:
    """Результат валидации адреса"""
    address: str
    is_valid: bool
    address_type: str
    network: str
    format_description: str
    technical_details: dict
    error_message: Optional[str] = None

class BitcoinAddressValidator:
    """Профессиональный валидатор Bitcoin адресов"""
    
    def __init__(self):
        self.console = Console()
        self.validation_history = []
        
        # Bech32 алфавит
        self.BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        
        # Описания типов адресов
        self.ADDRESS_DESCRIPTIONS = {
            'P2PKH': 'Pay to Public Key Hash (Legacy)',
            'P2SH': 'Pay to Script Hash (Multisig/SegWit)',
            'P2WPKH': 'Pay to Witness Public Key Hash (SegWit v0)',
            'P2WSH': 'Pay to Witness Script Hash (SegWit v0)',
            'P2TR': 'Pay to Taproot (SegWit v1)',
        }
    
    def show_welcome(self):
        """Красивое приветствие"""
        welcome_text = """
        🛡️  BITCOIN ADDRESS VALIDATOR  🛡️
        
        Профессиональный инструмент для валидации
        всех типов Bitcoin адресов
        
        ✅ Legacy (P2PKH, P2SH)
        ✅ SegWit v0 (Bech32)  
        ✅ SegWit v1 (Taproot)
        ✅ Testnet поддержка
        """
        
        panel = Panel(
            welcome_text,
            style="bold blue",
            border_style="bright_blue",
            title="Добро пожаловать!",
            title_align="center"
        )
        self.console.print(panel)
    
    def validate_legacy(self, address: str) -> ValidationResult:
        """Валидация Legacy адресов (P2PKH и P2SH)"""
        try:
            # Декодируем Base58Check
            decoded = base58.b58decode(address)
            
            # Проверяем длину (21 байт + 4 байта контрольная сумма)
            if len(decoded) != 25:
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверная длина адреса",
                    technical_details={"decoded_length": len(decoded)},
                    error_message="Длина должна быть 25 байт"
                )
            
            # Извлекаем компоненты
            version_byte = decoded[0]
            hash160 = decoded[1:21]
            checksum_received = decoded[21:25]
            
            # Вычисляем контрольную сумму
            payload = decoded[:-4]
            hash_once = hashlib.sha256(payload).digest()
            hash_twice = hashlib.sha256(hash_once).digest()
            checksum_calculated = hash_twice[:4]
            
            # Проверяем контрольную сумму
            if checksum_received != checksum_calculated:
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверная контрольная сумма",
                    technical_details={
                        "checksum_received": checksum_received.hex(),
                        "checksum_calculated": checksum_calculated.hex()
                    },
                    error_message="Контрольная сумма не совпадает"
                )
            
            # Определяем тип и сеть по version byte
            address_info = self._get_address_type_by_version(version_byte)
            
            return ValidationResult(
                address=address,
                is_valid=True,
                address_type=address_info['type'],
                network=address_info['network'],
                format_description=self.ADDRESS_DESCRIPTIONS[address_info['type']],
                technical_details={
                    "version_byte": f"0x{version_byte:02x}",
                    "hash160": hash160.hex(),
                    "checksum": checksum_received.hex()
                }
            )
            
        except Exception as e:
            return ValidationResult(
                address=address,
                is_valid=False,
                address_type="INVALID",
                network="unknown",
                format_description="Ошибка декодирования",
                technical_details={},
                error_message=str(e)
            )
    
    def _get_address_type_by_version(self, version_byte: int) -> dict:
        """Определяет тип адреса по version byte"""
        version_map = {
            0x00: {'type': 'P2PKH', 'network': 'mainnet'},
            0x05: {'type': 'P2SH', 'network': 'mainnet'},
            0x6f: {'type': 'P2PKH', 'network': 'testnet'},
            0xc4: {'type': 'P2SH', 'network': 'testnet'},
        }
        
        return version_map.get(version_byte, {'type': 'UNKNOWN', 'network': 'unknown'})
    
    def _bech32_polymod(self, values: List[int]) -> int:
        """Вычисляет Bech32 polymod"""
        generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
        chk = 1
        for value in values:
            top = chk >> 25
            chk = (chk & 0x1ffffff) << 5 ^ value
            for i in range(5):
                chk ^= generator[i] if ((top >> i) & 1) else 0
        return chk
    
    def _bech32_hrp_expand(self, hrp: str) -> List[int]:
        """Расширяет HRP для Bech32"""
        return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]
    
    def _bech32_verify_checksum(self, hrp: str, data: List[int], const: int) -> bool:
        """Проверяет Bech32 контрольную сумму"""
        return self._bech32_polymod(self._bech32_hrp_expand(hrp) + data) == const
    
    def validate_bech32(self, address: str) -> ValidationResult:
        """Валидация Bech32/Bech32m адресов (SegWit)"""
        try:
            address = address.lower()
            
            # Находим разделитель
            pos = address.rfind('1')
            if pos < 1 or pos + 7 > len(address):
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверный формат Bech32",
                    technical_details={},
                    error_message="Неверная позиция разделителя '1'"
                )
            
            hrp = address[:pos]
            data_part = address[pos+1:]
            
            # Проверяем HRP
            if hrp not in ['bc', 'tb']:
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверный HRP",
                    technical_details={"hrp": hrp},
                    error_message=f"Неподдерживаемый HRP: {hrp}"
                )
            
            # Конвертируем символы в числа
            if not all(c in self.BECH32_CHARSET for c in data_part):
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверные символы",
                    technical_details={},
                    error_message="Присутствуют недопустимые символы"
                )
            
            data = [self.BECH32_CHARSET.find(x) for x in data_part]
            
            # Проверяем контрольную сумму (Bech32 или Bech32m)
            is_bech32 = self._bech32_verify_checksum(hrp, data, 1)
            is_bech32m = self._bech32_verify_checksum(hrp, data, 0x2bc830a3)
            
            if not (is_bech32 or is_bech32m):
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Неверная контрольная сумма",
                    technical_details={},
                    error_message="Контрольная сумма Bech32 не прошла проверку"
                )
            
            # Извлекаем witness версию и программу
            if len(data) < 7:  # минимум 1 + 6 для контрольной суммы
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Слишком короткие данные",
                    technical_details={},
                    error_message="Недостаточно данных в адресе"
                )
            
            witness_version = data[0]
            program_5bit = data[1:-6]  # убираем контрольную сумму
            
            # Конвертируем из 5-битного в 8-битное представление
            program_8bit = self._convert_bits(program_5bit, 5, 8, False)
            if program_8bit is None:
                return ValidationResult(
                    address=address,
                    is_valid=False,
                    address_type="INVALID",
                    network="unknown",
                    format_description="Ошибка конвертации битов",
                    technical_details={},
                    error_message="Не удалось сконвертировать witness program"
                )
            
            # Валидируем witness program
            validation_result = self._validate_witness_program(witness_version, program_8bit, hrp)
            validation_result.address = address
            
            return validation_result
            
        except Exception as e:
            return ValidationResult(
                address=address,
                is_valid=False,
                address_type="INVALID",
                network="unknown",
                format_description="Ошибка обработки Bech32",
                technical_details={},
                error_message=str(e)
            )
    
    def _convert_bits(self, data: List[int], frombits: int, tobits: int, pad: bool) -> Optional[List[int]]:
        """Конвертирует между битовыми представлениями"""
        acc = 0
        bits = 0
        ret = []
        maxv = (1 << tobits) - 1
        max_acc = (1 << (frombits + tobits - 1)) - 1
        
        for value in data:
            if value < 0 or (value >> frombits):
                return None
            acc = ((acc << frombits) | value) & max_acc
            bits += frombits
            while bits >= tobits:
                bits -= tobits
                ret.append((acc >> bits) & maxv)
        
        if pad:
            if bits:
                ret.append((acc << (tobits - bits)) & maxv)
        elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
            return None
        
        return ret
    
    def _validate_witness_program(self, version: int, program: List[int], hrp: str) -> ValidationResult:
        """Валидирует witness program"""
        network = "mainnet" if hrp == "bc" else "testnet"
        program_bytes = bytes(program)
        
        if version == 0:
            # SegWit v0
            if len(program) == 20:
                address_type = "P2WPKH"
            elif len(program) == 32:
                address_type = "P2WSH"
            else:
                return ValidationResult(
                    address="",
                    is_valid=False,
                    address_type="INVALID",
                    network=network,
                    format_description="Неверная длина для SegWit v0",
                    technical_details={"program_length": len(program)},
                    error_message=f"SegWit v0 требует 20 или 32 байта, получено {len(program)}"
                )
        elif version == 1:
            # SegWit v1 (Taproot)
            if len(program) == 32:
                address_type = "P2TR"
            else:
                return ValidationResult(
                    address="",
                    is_valid=False,
                    address_type="INVALID",
                    network=network,
                    format_description="Неверная длина для Taproot",
                    technical_details={"program_length": len(program)},
                    error_message=f"Taproot требует 32 байта, получено {len(program)}"
                )
        else:
            address_type = f"SEGWIT_V{version}"
        
        return ValidationResult(
            address="",
            is_valid=True,
            address_type=address_type,
            network=network,
            format_description=self.ADDRESS_DESCRIPTIONS.get(address_type, f"SegWit версия {version}"),
            technical_details={
                "witness_version": version,
                "program": program_bytes.hex(),
                "program_length": len(program)
            }
        )
    
    def validate_address(self, address: str) -> ValidationResult:
        """Главная функция валидации"""
        address = address.strip()
        
        if not address:
            return ValidationResult(
                address="",
                is_valid=False,
                address_type="INVALID",
                network="unknown",
                format_description="Пустой адрес",
                technical_details={},
                error_message="Адрес не может быть пустым"
            )
        
        # Определяем тип по префиксу
        if address.startswith('1'):
            return self.validate_legacy(address)
        elif address.startswith('3'):
            return self.validate_legacy(address)
        elif address.startswith(('bc1', 'tb1')):
            return self.validate_bech32(address)
        else:
            return ValidationResult(
                address=address,
                is_valid=False,
                address_type="INVALID",
                network="unknown",
                format_description="Неизвестный формат",
                technical_details={"prefix": address[:4]},
                error_message=f"Неподдерживаемый префикс: {address[:10]}..."
            )
    
    def display_result(self, result: ValidationResult):
        """Красивое отображение результата"""
        # Определяем цветовую схему
        if result.is_valid:
            status_style = "green"
            status_icon = "✅"
            border_style = "bright_green"
        else:
            status_style = "red"
            status_icon = "❌"
            border_style = "bright_red"
        
        # Создаем таблицу с информацией
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Поле", style="bold cyan", width=18)
        table.add_column("Значение", style="white")
        
        table.add_row("Статус", f"[{status_style}]{status_icon} {'ВАЛИДНЫЙ' if result.is_valid else 'НЕВАЛИДНЫЙ'}[/{status_style}]")
        table.add_row("Адрес", f"[bold white]{result.address}[/bold white]")
        table.add_row("Тип", f"[bold yellow]{result.address_type}[/bold yellow]")
        table.add_row("Сеть", f"[bold blue]{result.network.upper()}[/bold blue]")
        table.add_row("Описание", f"[dim white]{result.format_description}[/dim white]")
        
        # Добавляем технические детали
        if result.technical_details:
            table.add_row("", "")  # Пустая строка
            for key, value in result.technical_details.items():
                display_key = key.replace('_', ' ').title()
                table.add_row(display_key, f"[dim]{value}[/dim]")
        
        if result.error_message:
            table.add_row("Ошибка", f"[red]{result.error_message}[/red]")
        
        # Создаем панель
        panel = Panel(
            table,
            title="🛡️ Результат валидации",
            border_style=border_style,
            padding=(1, 1)
        )
        
        self.console.print(panel)
    
    def show_statistics(self):
        """Показать статистику сессии"""
        if not self.validation_history:
            self.console.print("[yellow]📊 Нет данных для статистики[/yellow]")
            return
        
        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        invalid = total - valid
        
        # Статистика по типам
        type_counts = {}
        for result in self.validation_history:
            addr_type = result.address_type
            type_counts[addr_type] = type_counts.get(addr_type, 0) + 1
        
        # Главная статистика
        stats_table = Table(title="📊 Статистика валидации", show_header=True, header_style="bold blue")
        stats_table.add_column("Метрика", style="cyan")
        stats_table.add_column("Значение", justify="right")
        stats_table.add_column("Процент", justify="right", style="yellow")
        
        stats_table.add_row("Всего проверено", str(total), "100%")
        stats_table.add_row("✅ Валидных", str(valid), f"{valid/total*100:.1f}%")
        stats_table.add_row("❌ Невалидных", str(invalid), f"{invalid/total*100:.1f}%")
        
        self.console.print(stats_table)
        
        # Статистика по типам
        if type_counts:
            types_table = Table(title="📋 Распределение по типам", show_header=True, header_style="bold green")
            types_table.add_column("Тип адреса", style="cyan")
            types_table.add_column("Количество", justify="right")
            types_table.add_column("Процент", justify="right", style="yellow")
            
            for addr_type, count in sorted(type_counts.items()):
                types_table.add_row(addr_type, str(count), f"{count/total*100:.1f}%")
            
            self.console.print(types_table)
    
    def batch_validate(self, addresses: List[str]) -> List[ValidationResult]:
        """Пакетная валидация с прогресс-баром"""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Валидация адресов...", total=len(addresses))
            
            for address in addresses:
                result = self.validate_address(address)
                results.append(result)
                progress.update(task, advance=1)
        
        return results
    
    def interactive_mode(self):
        """Интерактивный режим"""
        self.show_welcome()
        
        self.console.print("\n[bold green]🚀 Добро пожаловать в интерактивный режим![/bold green]")
        self.console.print("\nДоступные команды:")
        self.console.print("  📝 [cyan]Введите Bitcoin адрес[/cyan] - валидация одного адреса")
        self.console.print("  📦 [cyan]batch[/cyan] - пакетная валидация")
        self.console.print("  📊 [cyan]stats[/cyan] - статистика сессии")
        self.console.print("  ❓ [cyan]help[/cyan] - справка")
        self.console.print("  🚪 [cyan]quit[/cyan] - выход")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]Команда или адрес[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[green]👋 Спасибо за использование! До свидания![/green]")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                
                elif user_input.lower() == 'batch':
                    self.batch_mode()
                
                else:
                    # Валидируем адрес
                    result = self.validate_address(user_input)
                    self.validation_history.append(result)
                    self.display_result(result)
                    
            except KeyboardInterrupt:
                self.console.print("\n[green]👋 Программа завершена![/green]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ Ошибка: {e}[/red]")
    
    def batch_mode(self):
        """Режим пакетной валидации"""
        self.console.print("\n[bold yellow]📦 Пакетная валидация[/bold yellow]")
        self.console.print("Введите адреса (по одному в строке).")
        self.console.print("Пустая строка для завершения ввода.\n")
        
        addresses = []
        line_num = 1
        
        while True:
            try:
                address = input(f"[{line_num}] ").strip()
                if not address:
                    break
                addresses.append(address)
                line_num += 1
            except KeyboardInterrupt:
                break
        
        if not addresses:
            self.console.print("[yellow]⚠️  Нет адресов для валидации[/yellow]")
            return
        
        self.console.print(f"\n🔍 Начинаю валидацию {len(addresses)} адресов...")
        
        results = self.batch_validate(addresses)
        self.validation_history.extend(results)
        
        # Показываем результаты
        self.console.print("\n📋 Результаты:")
        for i, result in enumerate(results, 1):
            self.console.print(f"\n[bold]Адрес #{i}:[/bold]")
            self.display_result(result)
        
        # Итоговая статистика
        valid_count = sum(1 for r in results if r.is_valid)
        self.console.print(f"\n✅ Итого: {valid_count}/{len(results)} валидных адресов")
    
    def show_help(self):
        """Показать справку"""
        help_content = """
[bold cyan]🛡️ Bitcoin Address Validator - Справка[/bold cyan]

[bold yellow]Поддерживаемые форматы:[/bold yellow]
• [green]Legacy P2PKH[/green]: начинаются с '1' 
  Пример: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
• [green]Legacy P2SH[/green]: начинаются с '3'
  Пример: 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy  
• [green]SegWit v0 P2WPKH[/green]: начинаются с 'bc1q' (20 байт)
  Пример: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
• [green]SegWit v0 P2WSH[/green]: начинаются с 'bc1q' (32 байта)
• [green]SegWit v1 Taproot[/green]: начинаются с 'bc1p'
  Пример: bc1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr

[bold yellow]Что проверяется:[/bold yellow]
✅ Формат адреса
✅ Контрольная сумма (Base58Check, Bech32)
✅ Длина адреса
✅ Тип адреса и сеть
✅ Witness version для SegWit

[bold yellow]Команды:[/bold yellow]
• [cyan]batch[/cyan] - пакетная валидация нескольких адресов
• [cyan]stats[/cyan] - статистика текущей сессии
• [cyan]help[/cyan] - эта справка
• [cyan]quit[/cyan] - выход из программы
        """
        
        panel = Panel(
            help_content,
            title="📚 Справка по использованию",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)

def main():
    """Главная функция"""
    validator = BitcoinAddressValidator()
    
    if len(sys.argv) > 1:
        # Режим командной строки
        address = sys.argv[1]
        result = validator.validate_address(address)
        validator.display_result(result)
        
        # JSON вывод если запрошено
        if len(sys.argv) > 2 and sys.argv[2] == '--json':
            result_dict = {
                'address': result.address,
                'is_valid': result.is_valid,
                'address_type': result.address_type,
                'network': result.network,
                'format_description': result.format_description,
                'technical_details': result.technical_details,
                'error_message': result.error_message,
                'timestamp': datetime.now().isoformat()
            }
            print(json.dumps(result_dict, indent=2))
    else:
        # Интерактивный режим
        validator.interactive_mode()

if __name__ == "__main__":
    main()
