"""
CARC Parser - Parses Claim Adjustment Reason Codes from X12.org CSV

Handles the official X12 CARC format with multi-line descriptions,
start/stop dates, and modification history.
"""

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class CARCCode:
    """Represents a single CARC (Claim Adjustment Reason Code)."""
    code: str
    description: str
    start_date: Optional[datetime] = None
    stop_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    notes: Optional[str] = None
    is_active: bool = True
    
    # Group code associations (CO, PR, OA, PI, CR)
    typical_group_codes: list = None
    
    def __post_init__(self):
        if self.typical_group_codes is None:
            self.typical_group_codes = self._infer_group_codes()
    
    def _infer_group_codes(self) -> list:
        """Infer typical group codes based on code description."""
        desc_lower = self.description.lower()
        
        # Patient responsibility indicators
        if any(term in desc_lower for term in ['deductible', 'coinsurance', 'copay', 'patient responsibility']):
            return ['PR']
        
        # Contractual obligation indicators
        if any(term in desc_lower for term in ['contractual', 'fee schedule', 'maximum allowable', 'contracted']):
            return ['CO']
        
        # Other adjustments
        if any(term in desc_lower for term in ['prior payer', 'duplicate', 'other payer']):
            return ['OA']
        
        # Default to CO (most common)
        return ['CO']


class CARCParser:
    """Parser for X12.org CARC CSV files."""
    
    # Common CARC codes with typical frequency weights (for distribution)
    # Based on industry statistics for behavioral health
    FREQUENCY_WEIGHTS = {
        '16': 0.15,   # Missing/invalid info - very common
        '18': 0.08,   # Duplicate claim
        '22': 0.06,   # Coordination of benefits
        '29': 0.07,   # Timely filing
        '45': 0.10,   # Fee schedule exceeded
        '50': 0.08,   # Medical necessity
        '96': 0.06,   # Non-covered charge
        '97': 0.05,   # Bundled service
        '197': 0.09,  # Prior auth missing
        '4': 0.04,    # Procedure/modifier inconsistent
        '5': 0.03,    # Procedure/place of service inconsistent
        '27': 0.03,   # Coverage terminated
        '31': 0.02,   # Patient not identified
        '109': 0.02,  # Wrong payer
        '167': 0.02,  # Diagnosis not covered
        # Remaining weight distributed among other codes
    }
    
    def __init__(self, csv_path: str = None):
        """
        Initialize CARC parser.
        
        Args:
            csv_path: Path to CARC CSV file from X12.org
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self.codes: dict[str, CARCCode] = {}
        self._active_codes: list[str] = []
        
        if self.csv_path and self.csv_path.exists():
            self.parse()
    
    def parse(self, csv_path: str = None) -> dict[str, CARCCode]:
        """
        Parse CARC CSV file.
        
        The X12 format has multi-line descriptions with dates embedded.
        Format: Code,Description where Description may contain:
        - Main description text
        - Start: MM/DD/YYYY
        - Last Modified: MM/DD/YYYY
        - Stop: MM/DD/YYYY
        - Notes: additional info
        """
        if csv_path:
            self.csv_path = Path(csv_path)
        
        if not self.csv_path or not self.csv_path.exists():
            raise FileNotFoundError(f"CARC CSV not found: {self.csv_path}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            current_code = None
            current_desc_lines = []
            
            for row in reader:
                if not row:
                    continue
                
                # Check if this is a new code line (starts with a code value)
                if row[0] and self._is_code_line(row[0]):
                    # Save previous code if exists
                    if current_code:
                        self._save_code(current_code, current_desc_lines)
                    
                    # Start new code
                    current_code = row[0].strip()
                    current_desc_lines = [row[1] if len(row) > 1 else '']
                else:
                    # Continuation of description
                    if current_code and row:
                        current_desc_lines.append(row[0] if row[0] else '')
            
            # Save last code
            if current_code:
                self._save_code(current_code, current_desc_lines)
        
        # Build active codes list
        self._active_codes = [code for code, carc in self.codes.items() if carc.is_active]
        
        return self.codes
    
    def _is_code_line(self, value: str) -> bool:
        """Check if this is a new code line (numeric or alphanumeric like A1, B7, P1)."""
        value = value.strip()
        # Match numeric codes (1-999) or letter+number codes (A1, B7, P1, etc.)
        return bool(re.match(r'^(\d+|[A-Z]\d+|[A-Z][A-Z]\d*)$', value))
    
    def _save_code(self, code: str, desc_lines: list):
        """Parse description lines and save CARC code."""
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
        
        # Determine if active (no stop date or stop date in future)
        is_active = stop_date is None or stop_date > datetime.now()
        
        self.codes[code] = CARCCode(
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
        """Return list of currently active CARC codes."""
        return self._active_codes.copy()
    
    def get_code(self, code: str) -> Optional[CARCCode]:
        """Get a specific CARC code."""
        return self.codes.get(code)
    
    def get_weighted_codes(self) -> dict[str, float]:
        """
        Return CARC codes with frequency weights for sampling.
        
        Returns dict of {code: weight} for active codes.
        Commonly denied codes get higher weights.
        """
        weights = {}
        active = self.get_active_codes()
        
        # Assign known weights
        total_known_weight = 0
        for code in active:
            if code in self.FREQUENCY_WEIGHTS:
                weights[code] = self.FREQUENCY_WEIGHTS[code]
                total_known_weight += self.FREQUENCY_WEIGHTS[code]
        
        # Distribute remaining weight among other active codes
        remaining_weight = max(0, 1.0 - total_known_weight)
        other_codes = [c for c in active if c not in self.FREQUENCY_WEIGHTS]
        
        if other_codes:
            per_code_weight = remaining_weight / len(other_codes)
            for code in other_codes:
                weights[code] = per_code_weight
        
        return weights
    
    def sample_code(self, rng=None) -> str:
        """
        Sample a CARC code according to realistic distribution.
        
        Args:
            rng: Random number generator (numpy.random.Generator or None)
        
        Returns:
            Sampled CARC code string
        """
        import random
        
        weights = self.get_weighted_codes()
        codes = list(weights.keys())
        probs = list(weights.values())
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        
        if rng is not None:
            # Use numpy RNG if provided
            return rng.choice(codes, p=probs)
        else:
            # Use stdlib random
            return random.choices(codes, weights=probs, k=1)[0]
    
    def get_statistics(self) -> dict:
        """Return statistics about parsed CARC codes."""
        active = [c for c in self.codes.values() if c.is_active]
        inactive = [c for c in self.codes.values() if not c.is_active]
        
        return {
            'total_codes': len(self.codes),
            'active_codes': len(active),
            'inactive_codes': len(inactive),
            'codes_with_notes': sum(1 for c in self.codes.values() if c.notes),
            'group_distribution': self._count_group_codes()
        }
    
    def _count_group_codes(self) -> dict:
        """Count CARC codes by inferred group code."""
        groups = {'CO': 0, 'PR': 0, 'OA': 0, 'PI': 0, 'CR': 0, 'unknown': 0}
        for carc in self.codes.values():
            if carc.is_active and carc.typical_group_codes:
                groups[carc.typical_group_codes[0]] = groups.get(carc.typical_group_codes[0], 0) + 1
        return groups


if __name__ == '__main__':
    # Test parser
    import sys
    
    if len(sys.argv) > 1:
        parser = CARCParser(sys.argv[1])
        stats = parser.get_statistics()
        print(f"Parsed {stats['total_codes']} CARC codes")
        print(f"Active: {stats['active_codes']}, Inactive: {stats['inactive_codes']}")
        print(f"\nGroup distribution: {stats['group_distribution']}")
        
        # Sample some codes
        print("\nSample codes (weighted):")
        for _ in range(5):
            code = parser.sample_code()
            carc = parser.get_code(code)
            print(f"  {code}: {carc.description[:60]}...")
