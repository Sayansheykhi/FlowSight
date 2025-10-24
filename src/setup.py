#!/usr/bin/env python3
"""
Setup script for Order Flow Engine.
Creates necessary directories and validates configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'data', 'exports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import websockets
        import pandas
        import numpy
        import plotly
        import dash
        import yaml
        print("✓ All core dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def validate_config():
    """Validate configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        print("✓ Configuration file exists")
        return True
    else:
        print("✗ Configuration file not found")
        print("The engine will create a default config.yaml on first run")
        return False

def main():
    """Main setup function."""
    print("Order Flow Engine Setup")
    print("=" * 30)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Check dependencies
    print("\nChecking dependencies...")
    deps_ok = check_dependencies()
    
    # Validate config
    print("\nValidating configuration...")
    config_ok = validate_config()
    
    print("\n" + "=" * 30)
    if deps_ok:
        print("✓ Setup complete! You can now run:")
        print("  python main.py")
    else:
        print("✗ Setup incomplete. Please install dependencies first:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
