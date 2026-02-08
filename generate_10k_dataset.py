#!/usr/bin/env python3
"""
Generate 10K Synthetic X12 835 Claims Dataset

This script generates:
- 10,000 synthetic healthcare claims
- JSON and CSV formats for ML training
- Labels file with denial ground truth
- X12 835 EDI files for validation research
- Statistics for IEEE paper reporting

Output: synthera835_output_10k/
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthera835 import ERA835Generator

def main():
    print("=" * 60)
    print("SynthERA-835: Generating 10,000 Claim Dataset")
    print("=" * 60)
    
    # Configure output directory
    output_dir = Path(__file__).parent.parent / "synthera835_output_10k"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator with fixed seed for reproducibility
    generator = ERA835Generator(
        denial_rate=0.11,  # 11% industry average denial rate
        seed=42  # Fixed seed for reproducible research
    )
    
    print(f"\nConfiguration:")
    print(f"  Output: {output_dir}")
    print(f"  Denial Rate: 11%")
    print(f"  Seed: 42 (reproducible)")
    print(f"  Claims: 10,000")
    print()
    
    # Generate claims (skip individual EDI files for 10K to save disk)
    print("Generating claims...")
    claims = generator.generate_dataset(
        num_claims=10000,
        output_dir=str(output_dir),
        include_edi=False  # Skip 10K individual EDI files
    )
    
    # Print statistics
    stats = generator.get_statistics()
    
    total = stats['claims_generated']
    denied = stats['claims_denied']
    partial = stats['claims_partial']
    paid = stats['claims_paid']
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nTotal Claims: {total}")
    print(f"Denied: {denied} ({denied/total:.1%})")
    print(f"Partial: {partial} ({partial/total:.1%})")
    print(f"Paid: {paid} ({paid/total:.1%})")
    
    if 'category_distribution' in stats:
        print(f"\nDenial Categories:")
        for cat, count in sorted(stats['category_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {cat}: {count}")
    
    if 'carc_distribution' in stats:
        print(f"\nTop CARC Codes:")
        for carc, count in sorted(stats['carc_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  CARC {carc}: {count}")
    
    print(f"\nOutput Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}: {size_kb:.1f} KB")
    
    print("\n✓ Dataset ready for ML training!")
    print("=" * 60)

if __name__ == "__main__":
    main()
