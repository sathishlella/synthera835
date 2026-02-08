"""
SynthERA-835 Generator - Core X12 835 EDI Generator

Generates synthetic Electronic Remittance Advice (ERA) files
in X12 835 format for research and benchmarking purposes.
"""

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from .carc_parser import CARCParser
from .rarc_parser import RARCParser


@dataclass
class ClaimLine:
    """Represents a single claim service line."""
    line_number: int
    procedure_code: str  # CPT/HCPCS
    modifier: Optional[str] = None
    charge_amount: float = 0.0
    paid_amount: float = 0.0
    adjustment_amount: float = 0.0
    units: int = 1
    date_of_service: datetime = None
    
    # Denial info
    carc_code: Optional[str] = None
    rarc_code: Optional[str] = None
    group_code: str = 'CO'  # CO, PR, OA, PI, CR
    
    # Labels for ML
    is_denied: bool = False
    denial_category: Optional[str] = None  # eligibility, clinical, admin, etc.
    is_recoverable: bool = False


@dataclass
class Claim:
    """Represents a synthetic healthcare claim."""
    claim_id: str
    patient_id: str
    provider_npi: str
    payer_id: str
    
    # Claim amounts
    total_charge: float = 0.0
    total_paid: float = 0.0
    patient_responsibility: float = 0.0
    
    # Claim metadata
    date_of_service: datetime = None
    claim_type: str = 'professional'  # professional, institutional
    specialty: str = 'behavioral_health'
    
    # Service lines
    lines: List[ClaimLine] = field(default_factory=list)
    
    # Claim-level adjustment
    claim_carc: Optional[str] = None
    claim_rarc: Optional[str] = None
    
    # Labels
    claim_status: str = 'paid'  # paid, denied, partial
    is_recoverable: bool = False
    recovery_action: Optional[str] = None


class ERA835Generator:
    """
    Generates synthetic X12 835 EDI files.
    
    Features:
    - Realistic CARC/RARC distributions from X12.org
    - Configurable denial rates
    - Behavioral health specialization
    - ML-ready labels
    """
    
    # Common CPT codes for behavioral health
    BEHAVIORAL_HEALTH_CPTS = {
        '90832': {'desc': 'Psychotherapy 30 min', 'base_charge': 85.00, 'time_min': 16, 'time_max': 37},
        '90834': {'desc': 'Psychotherapy 45 min', 'base_charge': 120.00, 'time_min': 38, 'time_max': 52},
        '90837': {'desc': 'Psychotherapy 60 min', 'base_charge': 160.00, 'time_min': 53, 'time_max': 999},
        '90847': {'desc': 'Family therapy with patient', 'base_charge': 140.00, 'time_min': 50, 'time_max': 50},
        '90846': {'desc': 'Family therapy without patient', 'base_charge': 140.00, 'time_min': 50, 'time_max': 50},
        '90853': {'desc': 'Group psychotherapy', 'base_charge': 50.00, 'time_min': 60, 'time_max': 90},
        '90791': {'desc': 'Psychiatric diagnostic evaluation', 'base_charge': 200.00, 'time_min': 60, 'time_max': 90},
        '90792': {'desc': 'Psychiatric evaluation with med services', 'base_charge': 250.00, 'time_min': 60, 'time_max': 90},
        '99213': {'desc': 'Office visit established low', 'base_charge': 95.00, 'time_min': 15, 'time_max': 29},
        '99214': {'desc': 'Office visit established moderate', 'base_charge': 130.00, 'time_min': 30, 'time_max': 44},
        '99215': {'desc': 'Office visit established high', 'base_charge': 180.00, 'time_min': 45, 'time_max': 59},
    }
    
    # Denial categories with recovery actions
    DENIAL_CATEGORIES = {
        'eligibility': {
            'carcs': ['27', '31', '32', '33', '177', '178', '180'],
            'recoverable': False,  # Usually not recoverable
            'action': 'verify_eligibility'
        },
        'authorization': {
            'carcs': ['197', '198', '38', '39'],
            'recoverable': True,  # Often recoverable with documentation
            'action': 'submit_auth'
        },
        'coding': {
            'carcs': ['4', '5', '6', '7', '8', '11', '16', '182', '199'],
            'recoverable': True,  # Recoverable with correction
            'action': 'correct_and_resubmit'
        },
        'medical_necessity': {
            'carcs': ['50', '56', '57', '150', '151', '152'],
            'recoverable': True,  # May recover with clinical docs
            'action': 'appeal_with_documentation'
        },
        'timely_filing': {
            'carcs': ['29'],
            'recoverable': False,  # Usually not recoverable
            'action': 'write_off'
        },
        'duplicate': {
            'carcs': ['18'],
            'recoverable': False,  # Check if actually duplicate
            'action': 'verify_original_payment'
        },
        'bundling': {
            'carcs': ['97', '59'],
            'recoverable': True,  # May recover with modifier
            'action': 'rebill_with_modifier'
        },
        'fee_schedule': {
            'carcs': ['45', '42'],
            'recoverable': False,  # Contractual
            'action': 'contractual_write_off'
        },
        'coordination': {
            'carcs': ['22', '23', '109'],
            'recoverable': True,  # Bill correct payer
            'action': 'bill_correct_payer'
        }
    }
    
    def __init__(
        self,
        carc_csv_path: str = None,
        rarc_csv_path: str = None,
        denial_rate: float = 0.11,  # 11% industry average
        seed: int = None
    ):
        """
        Initialize the ERA 835 generator.
        
        Args:
            carc_csv_path: Path to CARC CSV from X12.org
            rarc_csv_path: Path to RARC CSV from X12.org
            denial_rate: Target denial rate (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.denial_rate = denial_rate
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Load parsers
        self.carc_parser = CARCParser(carc_csv_path) if carc_csv_path else None
        self.rarc_parser = RARCParser(rarc_csv_path) if rarc_csv_path else None
        
        # Statistics
        self.stats = {
            'claims_generated': 0,
            'claims_denied': 0,
            'claims_paid': 0,
            'claims_partial': 0,
            'carc_distribution': {},
            'category_distribution': {}
        }
    
    def _generate_id(self, prefix: str, length: int = 10) -> str:
        """Generate a random ID."""
        chars = string.ascii_uppercase + string.digits
        suffix = ''.join(self.rng.choices(chars, k=length))
        return f"{prefix}{suffix}"
    
    def _generate_npi(self) -> str:
        """Generate a realistic NPI (10 digits starting with 1 or 2)."""
        first = self.rng.choice(['1', '2'])
        rest = ''.join(self.rng.choices(string.digits, k=9))
        return first + rest
    
    def _select_denial_category(self) -> str:
        """Select a denial category based on realistic distribution."""
        # Distribution based on industry data
        weights = {
            'coding': 0.25,
            'authorization': 0.20,
            'eligibility': 0.15,
            'medical_necessity': 0.12,
            'duplicate': 0.08,
            'timely_filing': 0.07,
            'bundling': 0.05,
            'coordination': 0.05,
            'fee_schedule': 0.03
        }
        categories = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(categories, weights=probs, k=1)[0]
    
    def _get_carc_for_category(self, category: str) -> str:
        """Get a CARC code for the given denial category."""
        if category in self.DENIAL_CATEGORIES:
            carcs = self.DENIAL_CATEGORIES[category]['carcs']
            # Filter to active codes if parser available
            if self.carc_parser:
                active = self.carc_parser.get_active_codes()
                carcs = [c for c in carcs if c in active] or carcs
            return self.rng.choice(carcs)
        
        # Fallback to weighted sampling
        if self.carc_parser:
            return self.carc_parser.sample_code()
        return '16'  # Default: missing info
    
    def _get_group_code(self, carc: str) -> str:
        """Determine the appropriate group code for a CARC."""
        if self.carc_parser:
            carc_obj = self.carc_parser.get_code(carc)
            if carc_obj and carc_obj.typical_group_codes:
                return carc_obj.typical_group_codes[0]
        
        # Default logic based on common patterns
        patient_resp_carcs = ['1', '2', '3', '66', '85', '100', '142', '187', '241']
        if carc in patient_resp_carcs:
            return 'PR'
        return 'CO'
    
    def generate_claim(
        self,
        claim_date: datetime = None,
        force_denial: bool = None,
        specialty: str = 'behavioral_health'
    ) -> Claim:
        """
        Generate a single synthetic claim.
        
        Args:
            claim_date: Date of service (default: random in past year)
            force_denial: Force denied/paid status (None = computed from features)
            specialty: Medical specialty
        
        Returns:
            Claim object with all fields populated
        """
        # Generate date if not provided
        if claim_date is None:
            days_ago = self.rng.randint(30, 365)
            claim_date = datetime.now() - timedelta(days=days_ago)
        
        # Create payer_id FIRST (needed for denial probability)
        payer_id = self._generate_id('PYR', 5)
        
        # Create claim
        claim = Claim(
            claim_id=self._generate_id('CLM'),
            patient_id=self._generate_id('PAT', 8),
            provider_npi=self._generate_npi(),
            payer_id=payer_id,
            date_of_service=claim_date,
            specialty=specialty
        )
        
        # Generate service lines (1-4 per claim)
        num_lines = self.rng.randint(1, 4)
        cpts = list(self.BEHAVIORAL_HEALTH_CPTS.keys())
        selected_cpts = self.rng.sample(cpts, min(num_lines, len(cpts)))
        
        for i, cpt in enumerate(selected_cpts, 1):
            cpt_info = self.BEHAVIORAL_HEALTH_CPTS[cpt]
            
            # Add some charge variance
            variance = self.rng.uniform(0.9, 1.1)
            charge = round(cpt_info['base_charge'] * variance, 2)
            
            line = ClaimLine(
                line_number=i,
                procedure_code=cpt,
                charge_amount=charge,
                units=1,
                date_of_service=claim_date
            )
            claim.lines.append(line)
            claim.total_charge += charge
        
        # ================================================================
        # REALISTIC DENIAL PROBABILITY (based on claim features)
        # ================================================================
        if force_denial is None:
            denial_prob = self._compute_denial_probability(claim)
            is_denied = self.rng.random() < denial_prob
        else:
            is_denied = force_denial
        
        if is_denied:
            self._apply_denial(claim)
        else:
            self._apply_payment(claim)
        
        # Update statistics
        self.stats['claims_generated'] += 1
        if claim.claim_status == 'denied':
            self.stats['claims_denied'] += 1
        elif claim.claim_status == 'partial':
            self.stats['claims_partial'] += 1
        else:
            self.stats['claims_paid'] += 1
        
        return claim
    
    def _compute_denial_probability(self, claim: Claim) -> float:
        """
        Compute denial probability based on claim features.
        
        Creates LEARNABLE patterns for ML models while maintaining
        overall denial rate around the target (11%).
        
        Risk factors:
        - Higher charges -> higher denial risk
        - Certain procedure codes -> higher denial risk
        - Certain payers (hash-based) -> higher denial risk
        - End of month/quarter -> higher denial risk (budget pressure)
        - More service lines -> higher denial risk (complexity)
        """
        base_rate = self.denial_rate  # 0.11 default
        
        # ================================================================
        # FACTOR 1: Charge amount risk
        # Higher charges = more scrutiny = higher denial risk
        # ================================================================
        charge = claim.total_charge
        if charge > 500:
            charge_factor = 0.08  # High charge claims +8% denial
        elif charge > 300:
            charge_factor = 0.04  # Medium-high +4%
        elif charge < 100:
            charge_factor = -0.03  # Low charge claims -3% denial
        else:
            charge_factor = 0.0
        
        # ================================================================
        # FACTOR 2: Procedure code risk
        # Some procedures have higher denial rates
        # ================================================================
        high_risk_procs = ['90792', '90791', '99215']  # Complex evaluations
        medium_risk_procs = ['90837', '90847', '90846']  # Long sessions/family
        low_risk_procs = ['99213', '90853']  # Routine visits, group
        
        procedure_factor = 0.0
        for line in claim.lines:
            if line.procedure_code in high_risk_procs:
                procedure_factor += 0.06  # +6% per high-risk proc
            elif line.procedure_code in medium_risk_procs:
                procedure_factor += 0.02  # +2% per medium-risk
            elif line.procedure_code in low_risk_procs:
                procedure_factor -= 0.02  # -2% per low-risk
        
        # Cap procedure factor
        procedure_factor = max(-0.05, min(0.15, procedure_factor))
        
        # ================================================================
        # FACTOR 3: Payer risk (hash-based for consistency)
        # Different payers have different denial rates
        # ================================================================
        payer_hash = hash(claim.payer_id) % 100
        if payer_hash < 20:
            payer_factor = 0.08  # "Tough" payers (20%) +8%
        elif payer_hash < 40:
            payer_factor = 0.03  # Moderate payers +3%
        elif payer_hash > 80:
            payer_factor = -0.04  # Easy payers -4%
        else:
            payer_factor = 0.0
        
        # ================================================================
        # FACTOR 4: Date/timing risk
        # End of quarter = budget pressure = more denials
        # ================================================================
        date = claim.date_of_service
        day_of_week = date.weekday()
        month = date.month
        
        # End of quarter months (March, June, Sept, Dec)
        if month in [3, 6, 9, 12]:
            quarter_factor = 0.03  # +3% denial in quarter-end
        else:
            quarter_factor = 0.0
        
        # Friday submissions get more denials (rushed processing)
        if day_of_week == 4:  # Friday
            day_factor = 0.02
        elif day_of_week == 0:  # Monday
            day_factor = -0.01  # Slightly better
        else:
            day_factor = 0.0
        
        # ================================================================
        # FACTOR 5: Complexity (number of lines)
        # More lines = more places for errors = more denials
        # ================================================================
        num_lines = len(claim.lines)
        if num_lines >= 4:
            complexity_factor = 0.05  # Many lines +5%
        elif num_lines >= 3:
            complexity_factor = 0.02  # Some lines +2%
        elif num_lines == 1:
            complexity_factor = -0.02  # Simple claim -2%
        else:
            complexity_factor = 0.0
        
        # ================================================================
        # COMBINE ALL FACTORS
        # ================================================================
        total_prob = (
            base_rate +
            charge_factor +
            procedure_factor +
            payer_factor +
            quarter_factor +
            day_factor +
            complexity_factor
        )
        
        # Clamp probability between 0.02 and 0.50
        final_prob = max(0.02, min(0.50, total_prob))
        
        return final_prob

    
    def _apply_denial(self, claim: Claim):
        """Apply denial to a claim."""
        # Select denial category
        category = self._select_denial_category()
        cat_info = self.DENIAL_CATEGORIES.get(category, {})
        
        # Update category stats
        self.stats['category_distribution'][category] = \
            self.stats['category_distribution'].get(category, 0) + 1
        
        # Get CARC and RARC
        carc = self._get_carc_for_category(category)
        group = self._get_group_code(carc)
        
        rarc = None
        if self.rarc_parser:
            rarc = self.rarc_parser.sample_rarc_for_carc(carc)
        
        # Update CARC stats
        self.stats['carc_distribution'][carc] = \
            self.stats['carc_distribution'].get(carc, 0) + 1
        
        # Decide if full or partial denial
        is_partial = self.rng.random() < 0.3  # 30% partial denials
        
        if is_partial:
            # Deny random subset of lines
            num_denied = self.rng.randint(1, max(1, len(claim.lines) - 1))
            denied_lines = self.rng.sample(claim.lines, num_denied)
            
            for line in claim.lines:
                if line in denied_lines:
                    line.is_denied = True
                    line.carc_code = carc
                    line.rarc_code = rarc
                    line.group_code = group
                    line.denial_category = category
                    line.is_recoverable = cat_info.get('recoverable', False)
                    line.adjustment_amount = line.charge_amount
                    line.paid_amount = 0.0
                else:
                    # Paid with some adjustment
                    adj_pct = self.rng.uniform(0.1, 0.3)
                    line.adjustment_amount = round(line.charge_amount * adj_pct, 2)
                    line.paid_amount = line.charge_amount - line.adjustment_amount
                    claim.total_paid += line.paid_amount
            
            claim.claim_status = 'partial'
        else:
            # Full denial
            for line in claim.lines:
                line.is_denied = True
                line.carc_code = carc
                line.rarc_code = rarc
                line.group_code = group
                line.denial_category = category
                line.is_recoverable = cat_info.get('recoverable', False)
                line.adjustment_amount = line.charge_amount
                line.paid_amount = 0.0
            
            claim.claim_status = 'denied'
        
        claim.claim_carc = carc
        claim.claim_rarc = rarc
        claim.is_recoverable = cat_info.get('recoverable', False)
        claim.recovery_action = cat_info.get('action')
    
    def _apply_payment(self, claim: Claim):
        """Apply payment to a claim (with possible contractual adjustments)."""
        for line in claim.lines:
            # Apply typical contractual adjustment (10-30%)
            adj_pct = self.rng.uniform(0.1, 0.3)
            line.adjustment_amount = round(line.charge_amount * adj_pct, 2)
            line.paid_amount = round(line.charge_amount - line.adjustment_amount, 2)
            line.carc_code = '45'  # Fee schedule adjustment
            line.group_code = 'CO'
            line.is_denied = False
            
            claim.total_paid += line.paid_amount
        
        # Possible patient responsibility
        if self.rng.random() < 0.4:  # 40% have patient responsibility
            pt_resp = self.rng.uniform(10, 50)
            claim.patient_responsibility = round(pt_resp, 2)
        
        claim.claim_status = 'paid'
    
    def generate_dataset(
        self,
        num_claims: int,
        output_dir: str = None,
        include_edi: bool = True
    ) -> List[Claim]:
        """
        Generate a dataset of synthetic claims.
        
        Args:
            num_claims: Number of claims to generate
            output_dir: Directory to save output files
            include_edi: Whether to generate X12 835 EDI files
        
        Returns:
            List of Claim objects
        """
        claims = []
        
        for _ in range(num_claims):
            claim = self.generate_claim()
            claims.append(claim)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON (for ML)
            self._save_claims_json(claims, output_path / 'claims.json')
            
            # Save as CSV (for analysis)
            self._save_claims_csv(claims, output_path / 'claims.csv')
            
            # Save labels
            self._save_labels(claims, output_path / 'labels.csv')
            
            # Generate EDI files
            if include_edi:
                edi_dir = output_path / 'edi'
                edi_dir.mkdir(exist_ok=True)
                for claim in claims:
                    edi_content = self.generate_edi_835(claim)
                    edi_path = edi_dir / f'{claim.claim_id}.835'
                    edi_path.write_text(edi_content)
            
            # Save statistics
            self._save_stats(output_path / 'statistics.json')
        
        return claims
    
    def _save_claims_json(self, claims: List[Claim], path: Path):
        """Save claims as JSON."""
        data = []
        for claim in claims:
            claim_dict = {
                'claim_id': claim.claim_id,
                'patient_id': claim.patient_id,
                'provider_npi': claim.provider_npi,
                'payer_id': claim.payer_id,
                'date_of_service': claim.date_of_service.isoformat(),
                'total_charge': claim.total_charge,
                'total_paid': claim.total_paid,
                'patient_responsibility': claim.patient_responsibility,
                'claim_status': claim.claim_status,
                'claim_carc': claim.claim_carc,
                'claim_rarc': claim.claim_rarc,
                'is_recoverable': claim.is_recoverable,
                'recovery_action': claim.recovery_action,
                'lines': []
            }
            for line in claim.lines:
                line_dict = {
                    'line_number': line.line_number,
                    'procedure_code': line.procedure_code,
                    'charge_amount': line.charge_amount,
                    'paid_amount': line.paid_amount,
                    'adjustment_amount': line.adjustment_amount,
                    'is_denied': line.is_denied,
                    'carc_code': line.carc_code,
                    'rarc_code': line.rarc_code,
                    'group_code': line.group_code,
                    'denial_category': line.denial_category,
                    'is_recoverable': line.is_recoverable
                }
                claim_dict['lines'].append(line_dict)
            data.append(claim_dict)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_claims_csv(self, claims: List[Claim], path: Path):
        """Save claims as flat CSV."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'claim_id', 'patient_id', 'provider_npi', 'payer_id',
                'date_of_service', 'line_number', 'procedure_code',
                'charge_amount', 'paid_amount', 'adjustment_amount',
                'claim_status', 'is_denied', 'carc_code', 'rarc_code',
                'group_code', 'denial_category', 'is_recoverable', 'recovery_action'
            ])
            
            for claim in claims:
                for line in claim.lines:
                    writer.writerow([
                        claim.claim_id, claim.patient_id, claim.provider_npi,
                        claim.payer_id, claim.date_of_service.strftime('%Y-%m-%d'),
                        line.line_number, line.procedure_code,
                        line.charge_amount, line.paid_amount, line.adjustment_amount,
                        claim.claim_status, line.is_denied, line.carc_code,
                        line.rarc_code, line.group_code, line.denial_category,
                        line.is_recoverable, claim.recovery_action
                    ])
    
    def _save_labels(self, claims: List[Claim], path: Path):
        """Save ML labels as CSV."""
        import csv
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'claim_id', 'is_denied', 'denial_category', 'is_recoverable', 'recovery_action'
            ])
            
            for claim in claims:
                is_denied = claim.claim_status in ['denied', 'partial']
                category = claim.lines[0].denial_category if is_denied else None
                writer.writerow([
                    claim.claim_id, is_denied, category,
                    claim.is_recoverable, claim.recovery_action
                ])
    
    def _save_stats(self, path: Path):
        """Save generation statistics."""
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def generate_edi_835(self, claim: Claim) -> str:
        """
        Generate X12 835 EDI content for a claim.
        
        This generates a simplified but valid 835 structure.
        """
        segments = []
        
        # ISA - Interchange Control Header
        timestamp = datetime.now()
        isa_date = timestamp.strftime('%y%m%d')
        isa_time = timestamp.strftime('%H%M')
        control_num = self._generate_id('', 9)
        
        segments.append(
            f"ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       "
            f"*{isa_date}*{isa_time}*^*00501*{control_num}*0*P*:~"
        )
        
        # GS - Functional Group Header
        segments.append(
            f"GS*HP*SENDER*RECEIVER*{timestamp.strftime('%Y%m%d')}*{timestamp.strftime('%H%M%S')}"
            f"*1*X*005010X221A1~"
        )
        
        # ST - Transaction Set Header
        segments.append("ST*835*0001*005010X221A1~")
        
        # BPR - Financial Information
        total_paid = claim.total_paid
        segments.append(
            f"BPR*I*{total_paid:.2f}*C*ACH*CTX*01*999999999*DA*1234567890*1234567891**01"
            f"*999999992*DA*9876543210*{timestamp.strftime('%Y%m%d')}~"
        )
        
        # TRN - Reassociation Trace Number
        segments.append(f"TRN*1*{control_num}*1234567890~")
        
        # DTM - Production Date
        segments.append(f"DTM*405*{timestamp.strftime('%Y%m%d')}~")
        
        # N1 - Payer Identification
        segments.append(f"N1*PR*SYNTHETIC PAYER*XV*{claim.payer_id}~")
        
        # N1 - Payee Identification
        segments.append(f"N1*PE*SYNTHETIC PROVIDER*XX*{claim.provider_npi}~")
        
        # CLP - Claim Payment Information
        status_code = '1' if claim.claim_status == 'paid' else ('2' if claim.claim_status == 'partial' else '4')
        segments.append(
            f"CLP*{claim.claim_id}*{status_code}*{claim.total_charge:.2f}*{total_paid:.2f}*"
            f"{claim.patient_responsibility:.2f}*12*{claim.claim_id}*11*1~"
        )
        
        # NM1 - Patient Name
        segments.append(f"NM1*QC*1*SYNTHETIC*PATIENT****MI*{claim.patient_id}~")
        
        # Service Lines
        for line in claim.lines:
            # SVC - Service Payment Information
            segments.append(
                f"SVC*HC:{line.procedure_code}*{line.charge_amount:.2f}*{line.paid_amount:.2f}**{line.units}~"
            )
            
            # DTM - Service Date
            segments.append(f"DTM*472*{claim.date_of_service.strftime('%Y%m%d')}~")
            
            # CAS - Claim Adjustment
            if line.adjustment_amount > 0:
                carc = line.carc_code or '45'
                group = line.group_code or 'CO'
                segments.append(f"CAS*{group}*{carc}*{line.adjustment_amount:.2f}~")
            
            # AMT - Service Supplemental Amount
            segments.append(f"AMT*B6*{line.paid_amount:.2f}~")
            
            # LQ - Remark Code
            if line.rarc_code:
                segments.append(f"LQ*HE*{line.rarc_code}~")
        
        # SE - Transaction Set Trailer
        segment_count = len(segments) + 1
        segments.append(f"SE*{segment_count}*0001~")
        
        # GE - Functional Group Trailer
        segments.append("GE*1*1~")
        
        # IEA - Interchange Control Trailer
        segments.append(f"IEA*1*{control_num}~")
        
        return '\n'.join(segments)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return generation statistics."""
        stats = self.stats.copy()
        
        # Calculate rates
        if stats['claims_generated'] > 0:
            stats['denial_rate'] = stats['claims_denied'] / stats['claims_generated']
            stats['partial_rate'] = stats['claims_partial'] / stats['claims_generated']
        
        return stats


if __name__ == '__main__':
    import sys
    
    # Test generator
    carc_path = sys.argv[1] if len(sys.argv) > 1 else None
    rarc_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    generator = ERA835Generator(
        carc_csv_path=carc_path,
        rarc_csv_path=rarc_path,
        denial_rate=0.11,
        seed=42
    )
    
    # Generate sample
    print("Generating 10 sample claims...")
    claims = generator.generate_dataset(10, output_dir='./sample_output')
    
    print(f"\nStatistics:")
    stats = generator.get_statistics()
    print(f"  Total claims: {stats['claims_generated']}")
    print(f"  Denied: {stats['claims_denied']}")
    print(f"  Partial: {stats['claims_partial']}")
    print(f"  Paid: {stats['claims_paid']}")
    print(f"\nCARC distribution: {stats['carc_distribution']}")
    print(f"Category distribution: {stats['category_distribution']}")
