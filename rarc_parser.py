"""
RARC Parser - Parses Remittance Advice Remark Codes from X12.org CSV

Handles the official X12 RARC format with multi-line descriptions,
start/stop dates, and modification history.
"""

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class RARCCode:
    """Represents a single RARC (Remittance Advice Remark Code)."""
    code: str
    description: str
    start_date: Optional[datetime] = None
    stop_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    notes: Optional[str] = None
    is_active: bool = True
    is_alert: bool = False  # True if this is an informational alert
    
    def __post_init__(self):
        # Detect if this is an alert code
        self.is_alert = self.description.lower().startswith('alert:')


class RARCParser:
    """Parser for X12.org RARC CSV files."""
    
    # RARC codes commonly paired with specific CARC codes
    # Based on CMS guidelines and common usage patterns
    CARC_RARC_PAIRS = {
        '16': ['MA130', 'N4', 'N26', 'M76', 'M77', 'N56'],  # Missing info
        '18': ['M86', 'MA67'],  # Duplicate
        '29': ['N64', 'N62'],  # Timely filing
        '45': ['N2', 'N13'],   # Fee schedule
        '50': ['M25', 'M26', 'N10'],  # Medical necessity
        '96': ['M15', 'N19', 'N20'],  # Non-covered
        '97': ['M15', 'M71'],  # Bundled
        '197': ['MA50', 'N54'],  # Prior auth
        '4': ['M69', 'N22'],   # Modifier issues
        '5': ['M77', 'N79'],   # Place of service
    }
    
    def __init__(self, csv_path: str = None):
        """
        Initialize RARC parser.
        
        Args:
            csv_path: Path to RARC CSV file from X12.org
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self.codes: dict[str, RARCCode] = {}
        self._active_codes: list[str] = []
        self._alert_codes: list[str] = []
        
        if self.csv_path and self.csv_path.exists():
            self.parse()
    
    def parse(self, csv_path: str = None) -> dict[str, RARCCode]:
        """
        Parse RARC CSV file.
        
        The X12 format has: RARC CODE, RARC CODE DESCRIPTION
        Descriptions may span multiple lines and contain dates.
        """
        if csv_path:
            self.csv_path = Path(csv_path)
        
        if not self.csv_path or not self.csv_path.exists():
            raise FileNotFoundError(f"RARC CSV not found: {self.csv_path}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            current_code = None
            current_desc_lines = []
            
            for row in reader:
                if not row:
                    continue
                
                # Check if this is a new code line (RARC codes start with M, MA, N)
                first_col = row[0].strip() if row[0] else ''
                if self._is_code_line(first_col):
                    # Save previous code if exists
                    if current_code:
                        self._save_code(current_code, current_desc_lines)
                    
                    # Start new code
                    current_code = first_col
                    current_desc_lines = [row[1] if len(row) > 1 else '']
                else:
                    # Continuation of description
                    if current_code:
                        current_desc_lines.append(first_col)
            
            # Save last code
            if current_code:
                self._save_code(current_code, current_desc_lines)
        
        # Build index lists
        self._active_codes = [code for code, rarc in self.codes.items() if rarc.is_active]
        self._alert_codes = [code for code, rarc in self.codes.items() if rarc.is_alert and rarc.is_active]
        
        return self.codes
    
    def _is_code_line(self, value: str) -> bool:
        """Check if this is a new RARC code line."""
        # RARC codes: M1-M999, MA1-MA999, N1-N999
        return bool(re.match(r'^(M\d+|MA\d+|N\d+)$', value))
    
    def _save_code(self, code: str, desc_lines: list):
        """Parse description lines and save RARC code."""
        full_text = '\n'.join(desc_lines)
        
        # Extract main description (before Start: date)
        desc_match = re.split(r'\nStart:', full_text, maxsplit=1)
        description = desc_match[0].strip().replace('\n', ' ')
        
        # Extract dates
        start_date = self._extract_date(full_text, r'Start:\s*(\d{2}/\d{2}/\d{4})')
        stop_date = self._extract_date(full_text, r'Stop:\s*(\d{2}/\d{2}/\d{4})')
        last_modified = self._extract_date(full_text, r'Last Modified:\s*(\d{2}/\d{2}/\d{4})')
        
        # Extract notes
        notes_match = re.search(r'Notes:\s*(.+?)(?:\n|$)', full_text)
        notes = notes_match.group(1).strip() if notes_match else None
        
        # Determine if active
        is_active = stop_date is None or stop_date > datetime.now()
        
        self.codes[code] = RARCCode(
            code=code,
            description=description,
            start_date=start_date,
            stop_date=stop_date,
            last_modified=last_modified,
            notes=notes,
            is_active=is_active
        )
    
    def _extract_date(self, text: str, pattern: str) -> Optional[datetime]:
        """Extract date from text using regex pattern."""
        match = re.search(pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(1), '%m/%d/%Y')
            except ValueError:
                return None
        return None
    
    def get_active_codes(self) -> list[str]:
        """Return list of currently active RARC codes."""
        return self._active_codes.copy()
    
    def get_alert_codes(self) -> list[str]:
        """Return list of active alert-type RARC codes."""
        return self._alert_codes.copy()
    
    def get_code(self, code: str) -> Optional[RARCCode]:
        """Get a specific RARC code."""
        return self.codes.get(code)
    
    def get_rarc_for_carc(self, carc_code: str) -> list[str]:
        """
        Get RARC codes commonly paired with a specific CARC code.
        
        Args:
            carc_code: CARC code to find RARC pairs for
        
        Returns:
            List of appropriate RARC codes
        """
        # Check predefined pairs
        if carc_code in self.CARC_RARC_PAIRS:
            return [r for r in self.CARC_RARC_PAIRS[carc_code] if r in self.codes and self.codes[r].is_active]
        
        # Return generic codes for unknown CARC
        generic_rarcs = ['N1', 'N59', 'M16']
        return [r for r in generic_rarcs if r in self.codes and self.codes[r].is_active]
    
    def sample_rarc_for_carc(self, carc_code: str, include_alert: bool = True, rng=None) -> Optional[str]:
        """
        Sample a RARC code appropriate for the given CARC code.
        
        Args:
            carc_code: The CARC code to pair with
            include_alert: Whether to possibly include an alert RARC
            rng: Random number generator
        
        Returns:
            Sampled RARC code or None
        """
        import random
        
        candidates = self.get_rarc_for_carc(carc_code)
        
        if include_alert and random.random() < 0.3:
            # 30% chance to add an alert code
            alerts = self.get_alert_codes()
            if alerts:
                candidates = candidates + [random.choice(alerts)]
        
        if not candidates:
            return None
        
        if rng is not None:
            return rng.choice(candidates)
        return random.choice(candidates)
    
    def get_statistics(self) -> dict:
        """Return statistics about parsed RARC codes."""
        active = [c for c in self.codes.values() if c.is_active]
        alerts = [c for c in active if c.is_alert]
        
        # Count by prefix
        m_codes = sum(1 for c in active if c.code.startswith('M') and not c.code.startswith('MA'))
        ma_codes = sum(1 for c in active if c.code.startswith('MA'))
        n_codes = sum(1 for c in active if c.code.startswith('N'))
        
        return {
            'total_codes': len(self.codes),
            'active_codes': len(active),
            'inactive_codes': len(self.codes) - len(active),
            'alert_codes': len(alerts),
            'by_prefix': {
                'M': m_codes,
                'MA': ma_codes,
                'N': n_codes
            }
        }


if __name__ == '__main__':
    # Test parser
    import sys
    
    if len(sys.argv) > 1:
        parser = RARCParser(sys.argv[1])
        stats = parser.get_statistics()
        print(f"Parsed {stats['total_codes']} RARC codes")
        print(f"Active: {stats['active_codes']}, Inactive: {stats['inactive_codes']}")
        print(f"Alert codes: {stats['alert_codes']}")
        print(f"\nBy prefix: {stats['by_prefix']}")
        
        # Test CARC-RARC pairing
        print("\nSample RARC for CARC 16 (missing info):")
        for _ in range(3):
            rarc = parser.sample_rarc_for_carc('16')
            if rarc:
                code_obj = parser.get_code(rarc)
                print(f"  {rarc}: {code_obj.description[:50]}...")
