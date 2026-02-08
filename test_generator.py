"""
Test script for SynthERA-835 Generator

Validates the generator works with official X12.org CARC/RARC files.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthera835 import ERA835Generator, CARCParser, RARCParser


def test_carc_parser():
    """Test CARC parser with official file."""
    print("=" * 60)
    print("TESTING CARC PARSER")
    print("=" * 60)
    
    carc_path = r"c:\Users\Sathish\Desktop\Velden_Research\Claim Adjustment Reason Codes(CARC).csv"
    
    parser = CARCParser(carc_path)
    stats = parser.get_statistics()
    
    print(f"✓ Parsed {stats['total_codes']} CARC codes")
    print(f"  - Active: {stats['active_codes']}")
    print(f"  - Inactive: {stats['inactive_codes']}")
    print(f"  - Codes with notes: {stats['codes_with_notes']}")
    print(f"  - Group distribution: {stats['group_distribution']}")
    
    # Test specific codes
    print("\n  Sample codes:")
    test_codes = ['16', '18', '29', '45', '50', '96', '197']
    for code in test_codes:
        carc = parser.get_code(code)
        if carc:
            status = "✓" if carc.is_active else "✗"
            print(f"    {status} CARC {code}: {carc.description[:50]}...")
    
    # Test weighted sampling
    print("\n  Sampled codes (10 samples):")
    samples = [parser.sample_code() for _ in range(10)]
    from collections import Counter
    sample_counts = Counter(samples)
    print(f"    {dict(sample_counts)}")
    
    return parser


def test_rarc_parser():
    """Test RARC parser with official file."""
    print("\n" + "=" * 60)
    print("TESTING RARC PARSER")
    print("=" * 60)
    
    rarc_path = r"c:\Users\Sathish\Desktop\Velden_Research\Remittance Advice Remark Codes(RARC).csv"
    
    parser = RARCParser(rarc_path)
    stats = parser.get_statistics()
    
    print(f"✓ Parsed {stats['total_codes']} RARC codes")
    print(f"  - Active: {stats['active_codes']}")
    print(f"  - Inactive: {stats['inactive_codes']}")
    print(f"  - Alert codes: {stats['alert_codes']}")
    print(f"  - By prefix: {stats['by_prefix']}")
    
    # Test CARC-RARC pairing
    print("\n  CARC-RARC pairs:")
    test_carcs = ['16', '50', '197']
    for carc in test_carcs:
        rarcs = parser.get_rarc_for_carc(carc)
        print(f"    CARC {carc} -> {rarcs[:3]}...")
    
    return parser


def test_generator():
    """Test full generator with CARC/RARC files."""
    print("\n" + "=" * 60)
    print("TESTING ERA 835 GENERATOR")
    print("=" * 60)
    
    carc_path = r"c:\Users\Sathish\Desktop\Velden_Research\Claim Adjustment Reason Codes(CARC).csv"
    rarc_path = r"c:\Users\Sathish\Desktop\Velden_Research\Remittance Advice Remark Codes(RARC).csv"
    
    generator = ERA835Generator(
        carc_csv_path=carc_path,
        rarc_csv_path=rarc_path,
        denial_rate=0.11,
        seed=42
    )
    
    # Generate sample claims
    print("\n  Generating 100 sample claims...")
    output_dir = r"c:\Users\Sathish\Desktop\Velden_Research\synthera835_output"
    claims = generator.generate_dataset(100, output_dir=output_dir, include_edi=True)
    
    stats = generator.get_statistics()
    
    print(f"\n✓ Generated {stats['claims_generated']} claims")
    print(f"  - Denied: {stats['claims_denied']} ({stats.get('denial_rate', 0)*100:.1f}%)")
    print(f"  - Partial: {stats['claims_partial']}")
    print(f"  - Paid: {stats['claims_paid']}")
    
    print(f"\n  CARC Distribution (top 5):")
    sorted_carcs = sorted(stats['carc_distribution'].items(), key=lambda x: -x[1])[:5]
    for carc, count in sorted_carcs:
        print(f"    CARC {carc}: {count}")
    
    print(f"\n  Denial Category Distribution:")
    for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    print(f"\n✓ Output saved to: {output_dir}")
    print(f"  - claims.json (for ML)")
    print(f"  - claims.csv (for analysis)")
    print(f"  - labels.csv (ground truth)")
    print(f"  - edi/*.835 (X12 835 files)")
    
    # Show sample EDI
    print("\n  Sample EDI (first claim):")
    sample_edi = generator.generate_edi_835(claims[0])
    for line in sample_edi.split('\n')[:10]:
        print(f"    {line}")
    print("    ...")
    
    return generator


def main():
    """Run all tests."""
    print("\n🔬 SynthERA-835 Generator Test Suite")
    print("=" * 60)
    
    try:
        carc_parser = test_carc_parser()
        rarc_parser = test_rarc_parser()
        generator = test_generator()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nGenerator is ready for research use!")
        print("Next steps:")
        print("  1. Generate larger dataset (10K+ claims)")
        print("  2. Train baseline models (RF, XGBoost)")
        print("  3. Run experiments")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
