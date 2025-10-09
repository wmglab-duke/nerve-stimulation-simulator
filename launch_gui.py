#!/usr/bin/env python3
"""
Launch script for the Nerve Stimulation Simulator GUI

This script launches the Streamlit web interface for the nerve stimulation simulator.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit GUI."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_dir = script_dir / "streamlit_gui"
    app_file = streamlit_dir / "app.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"❌ Error: Streamlit app not found at {app_file}")
        print("Please make sure the streamlit_gui directory and app.py file exist.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Installing requirements...")
        requirements_file = streamlit_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "matplotlib", "scipy"])
    
    print("🚀 Launching Nerve Stimulation Simulator GUI...")
    print("📱 The web interface will open in your default browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 GUI stopped by user")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
