#!/usr/bin/env python3
"""
NeuralFlix Setup Script
Run this to install dependencies and launch the app.
"""
import subprocess
import sys
import os

def install_packages():
    packages = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "requests>=2.28.0",
        "Pillow>=9.0.0",
        "plotly>=5.15.0",
    ]
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("All packages installed!\n")


def run_app():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Launching NeuralFlix...")
    print("Open your browser at: http://localhost:8501\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                    "--server.port=8501",
                    "--server.headless=false",
                    "--theme.base=dark"])


if __name__ == "__main__":
    try:
        import streamlit
    except ImportError:
        install_packages()
    run_app()
