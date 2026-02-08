#!/usr/bin/env python3
"""
Data Leakage Detection Script
==============================

Run this script to audit your dataset or pipeline for data leakage.

This is THE MOST IMPORTANT script in this project.
It teaches you to think like a senior ML engineer.

Usage:
    python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv
    python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv --introduce-leakage scaling
    python scripts/detect_leakage.py --help

Author: Senior ML Engineer
Date: February 2026
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.validators import DataLeakageDetector, deliberately_introduce_leakage
from src.data.loader import NVDADataLoader


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Data Leakage Forensic Audit for ML Pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic audit
  python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv
  
  # Educational mode: Introduce leakage to learn detection
  python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv --introduce-leakage scaling
  python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv --introduce-leakage future_lag
  
  # Save report
  python scripts/detect_leakage.py --data data/raw/NVDA_yfinance_clean.csv --output reports/leakage_report.json

Leakage Types (for educational mode):
  scaling          - Scale data before train-test split (COMMON MISTAKE)
  future_lag       - Create lag features with future information
  target_encoding  - Use target statistics as features
  
This tool helps you understand WHY data leakage causes production failures.
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV data file'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='Close',
        help='Target column name (default: Close)'
    )
    
    parser.add_argument(
        '--date-col',
        type=str,
        default='Date',
        help='Date column name (default: Date)'
    )
    
    parser.add_argument(
        '--introduce-leakage',
        type=str,
        choices=['scaling', 'future_lag', 'target_encoding'],
        help='EDUCATIONAL MODE: Deliberately introduce leakage to learn detection'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save report to JSON file'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.95,
        help='Correlation threshold for leakage warning (default: 0.95)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🔍 DATA LEAKAGE FORENSIC AUDIT 🔍                       ║
║                                                                              ║
║     "A model that cheats in development will fail in production"            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    # Load data
    try:
        if not args.quiet:
            print(f"\n📂 Loading data from: {args.data}")
        
        loader = NVDADataLoader(args.data, date_column=args.date_col)
        df, quality_report = loader.load_and_validate(verbose=not args.quiet)
        
        if not quality_report.is_valid:
            print("\n⚠️  WARNING: Data quality issues detected!")
            print("These may affect leakage detection. Consider fixing them first.")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've downloaded the data file:")
        print(f"   Place NVDA_yfinance_clean.csv in: {Path(args.data).parent}")
        return 1
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return 1
    
    # Educational mode: Introduce leakage
    if args.introduce_leakage:
        print("\n" + "="*80)
        print(f"⚠️  EDUCATIONAL MODE: Introducing {args.introduce_leakage.upper()} leakage")
        print("="*80)
        print("\nThis is for LEARNING purposes!")
        print("You will:")
        print("1. See how the leakage is introduced")
        print("2. Learn to detect it")
        print("3. Understand why it causes production failures")
        print("\n" + "="*80)
        
        df = deliberately_introduce_leakage(df, args.target, args.introduce_leakage)
    
    # Run leakage detection
    detector = DataLeakageDetector(
        df,
        target_col=args.target,
        date_col=args.date_col,
        threshold_correlation=args.correlation_threshold,
        verbose=not args.quiet
    )
    
    report = detector.run_full_audit()
    
    # Save report if requested
    if args.output:
        import json
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to dict
        report_dict = {
            'leakage_detected': report.leakage_detected,
            'severity': report.severity,
            'leakage_types': report.leakage_types,
            'details': report.details,
            'recommendations': report.recommendations,
            'timestamp': report.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {output_path}")
    
    # Educational summary
    if args.introduce_leakage and not args.quiet:
        print("\n" + "="*80)
        print("🎓 EDUCATIONAL SUMMARY")
        print("="*80)
        
        if report.leakage_detected:
            print(f"\n✅ SUCCESS! Leakage was detected!")
            print(f"   Type introduced: {args.introduce_leakage}")
            print(f"   Types detected: {', '.join(report.leakage_types)}")
            print("\n💡 What you learned:")
            print(f"   • How {args.introduce_leakage} leakage happens")
            print("   • How to detect it systematically")
            print("   • Why it makes models fail in production")
        else:
            print(f"\n⚠️  Leakage was introduced but not all types were detected!")
            print("   This might be a limitation of the detector or the data.")
            print("   In real projects, you'd need multiple detection strategies.")
        
        print("\n📚 Next steps:")
        print("   1. Try other leakage types: scaling, future_lag, target_encoding")
        print("   2. Train a model on clean vs leaky data")
        print("   3. Compare their production performance")
        print("   4. Understand the difference in metrics")
        
        print("\n" + "="*80)
    
    # Exit code based on detection
    if report.leakage_detected and report.severity in ['CRITICAL', 'HIGH']:
        print("\n🚨 CRITICAL LEAKAGE DETECTED - FIX BEFORE PROCEEDING!\n")
        return 2 if not args.introduce_leakage else 0  # Don't fail in educational mode
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
