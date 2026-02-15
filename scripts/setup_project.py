#!/usr/bin/env python3
"""
Project Setup Script
====================

Run this script ONCE after cloning the repository to set up your environment.

Usage:
    python scripts/setup_project.py

Author: Senior ML Engineer
Date: February 2026
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str):
    """Run a shell command with error handling"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🚀 NVIDIA STOCK ANALYSIS PROJECT SETUP 🚀                     ║
║                                                                              ║
║          Building Production-Ready ML Pipelines for Time-Series             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    print(f"\n📁 Project Root: {project_root.absolute()}")
    
    # Step 1: Check Python version
    print("\n" + "="*80)
    print("🐍 Checking Python Version")
    print("="*80)
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("❌ Python 3.10+ is required!")
        print("Please upgrade your Python version.")
        return False
    else:
        print("✅ Python version is compatible!")
    
    # Step 2: Create virtual environment
    venv_path = project_root / "venv"
    
    # Define venv python path explicitly
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"
    
    if not venv_path.exists():
        response = input("\n💡 Create virtual environment? (y/n): ")
        if response.lower() == 'y':
            run_command(
                f"python3 -m venv {venv_path}",
                "Creating virtual environment"
            )
            
            # Detect OS for activation command
            if sys.platform == "win32":
                activate_cmd = f"{venv_path}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_path}/bin/activate"
            
            print(f"\n💡 To activate virtual environment, run:")
            print(f"   {activate_cmd}")
    else:
        print("\n✅ Virtual environment already exists")
    
    # Step 3: Install dependencies
    response = input("\n💡 Install Python dependencies? (y/n): ")
    if response.lower() == 'y':
        # Prioritize installing into the venv we just checked/created
        if venv_python.exists():
            print(f"📦 Installing dependencies into virtual environment: {venv_python}")
            run_command(
                f"\"{venv_python}\" -m pip install -r {project_root}/requirements.txt",
                "Installing dependencies"
            )
            
            # Step 3.5: Register Jupyter Kernel (Fixes VS Code discovery issues)
            print("\n" + "="*80)
            print("⚙️  Registering Jupyter Kernel")
            print("="*80)
            run_command(
                f"\"{venv_python}\" -m ipykernel install --user --name=nvidia_stock_project --display-name \"Python (NVIDIA Stock Project)\"",
                "Registering kernel with Jupyter"
            )
    
    # Step 4: Initialize Git (if not already initialized)
    git_dir = project_root / ".git"
    if not git_dir.exists():
        response = input("\n💡 Initialize Git repository? (y/n): ")
        if response.lower() == 'y':
            run_command("git init", "Initializing Git repository")
            run_command("git add .", "Staging files")
            run_command(
                'git commit -m "Initial commit: Project structure"',
                "Creating initial commit"
            )
            print("\n💡 Git repository initialized!")
            print("   To connect to remote repository:")
            print("   git remote add origin <your-repo-url>")
            print("   git push -u origin main")
    else:
        print("\n✅ Git repository already initialized")
    
    # Step 5: Download data (if needed)
    data_file = project_root / "data" / "raw" / "NVDA_yfinance_clean.csv"
    
    if not data_file.exists():
        print("\n" + "="*80)
        print("📥 DATA DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print("\n🔗 Download NVDA stock data:")
        print("   1. Go to: https://www.kaggle.com/datasets/...")
        print("   2. Download NVDA_yfinance_clean.csv")
        print(f"   3. Place it in: {data_file.parent}")
        print("\n   OR use Kaggle API:")
        print("   kaggle datasets download -d <dataset-id> -p data/raw/")
    else:
        print(f"\n✅ Data file found: {data_file.name}")
    
    # Step 6: Run tests
    response = input("\n💡 Run initial tests? (y/n): ")
    if response.lower() == 'y':
        run_command(
            f"cd {project_root} && \"{sys.executable}\" -m pytest tests/ -v",
            "Running tests"
        )
    
    # Final instructions
    print("\n" + "="*80)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*80)
    
    print("\n📚 NEXT STEPS:")
    print("\n1. Activate your virtual environment:")
    if sys.platform == "win32":
        print(f"   {venv_path}\\Scripts\\activate")
    else:
        print(f"   source {venv_path}/bin/activate")
    
    print("\n2. Open Jupyter Lab:")
    print("   jupyter lab")
    
    print("\n3. Start with the notebooks in order:")
    print("   📓 notebooks/01_eda.ipynb")
    print("   📓 notebooks/02_feature_engineering.ipynb")
    print("   📓 notebooks/03_baseline_models.ipynb")
    print("   📓 notebooks/04_advanced_models.ipynb")
    print("   📓 notebooks/05_data_leakage_audit.ipynb  🔥 CRITICAL!")
    
    print("\n4. Or run the data leakage detector:")
    print("   python scripts/detect_leakage.py --help")
    
    print("\n" + "="*80)
    print("💡 TIP: Read README.md for detailed documentation")
    print("="*80)
    
    print("\n🚀 Happy learning! Become a 100x engineer! 🚀\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
