#!/usr/bin/env python3
"""
Launch script for the Nerve Stimulation Simulator GUI

This script launches the Streamlit web interface for the nerve stimulation simulator.
"""

import subprocess
import sys
import os
from pathlib import Path

# Try to launch Streamlit directly in-process
def launch_streamlit_in_process(app_file, port=8501):
    """Launch Streamlit directly without subprocess."""
    try:
        import streamlit.web.cli as stcli
        
        # Set up command line arguments for Streamlit
        sys.argv = [
            'streamlit',
            'run',
            str(app_file),
            '--server.port', str(port),
            '--server.address', 'localhost'
        ]
        
        print(f"✅ Launching Streamlit in-process...")
        stcli.main()
        return True
    except Exception as e:
        print(f"❌ In-process launch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Launch the Streamlit GUI."""
    print("🧠 Nerve Stimulation Simulator - Starting...")
    print("=" * 50)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_dir = script_dir / "streamlit_gui"
    app_file = streamlit_dir / "app.py"
    
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Streamlit directory: {streamlit_dir}")
    print(f"📄 App file: {app_file}")
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"❌ Error: Streamlit app not found at {app_file}")
        print("Please make sure the streamlit_gui directory and app.py file exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in script directory: {list(script_dir.iterdir())}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} found")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        print("⚠️  This is likely a PyInstaller packaging issue.")
        print("💡 The executable should include Streamlit - this shouldn't happen.")
        print("🔄 Attempting to continue anyway...")
    except Exception as e:
        print(f"❌ Unexpected error importing Streamlit: {e}")
        print("🔄 Attempting to continue anyway...")
    
    print("🚀 Launching Nerve Stimulation Simulator GUI...")
    print("📱 The web interface will open in your default browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Launch streamlit
    try:
        # First try in-process launch (best for PyInstaller)
        print("🔧 Attempting in-process Streamlit launch...")
        if launch_streamlit_in_process(app_file, port=8501):
            print("✅ Successfully launched Streamlit")
            return
        else:
            print("⚠️  In-process launch failed, trying subprocess approaches...")
        
        # Fallback: try using the bundled Python executable
        print(f"⚠️  Trying alternative launch method...")
        try:
            # Use the bundled Python to run streamlit module
            import subprocess
            cmd = [sys.executable, "-m", "streamlit", "run", str(app_file), 
                   "--server.port", "8501", "--server.address", "localhost"]
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Working directory: {script_dir}")
            
            subprocess.run(cmd, cwd=script_dir)
        except Exception as e:
            print(f"❌ Alternative launch failed: {e}")
            print("\n💡 TROUBLESHOOTING:")
            print("   - Make sure the executable includes Streamlit")
            print("   - Check that streamlit_gui/app.py exists")
            print("   - Try rebuilding with option 2 (console version) for debugging")
            input("\nPress Enter to exit...")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 GUI stopped by user")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find Python or Streamlit: {e}")
        print("Make sure Python and Streamlit are properly installed.")
        input("Press Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
