"""
VISCA Camera Video Tracking Application Launcher
Easy launcher for the PyQt6 version

Simply run: python launch_app.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check if all required packages are installed"""
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append('PyQt6')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        from visca_over_ip import Camera
    except ImportError:
        missing.append('visca_over_ip')
    
    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print("\nThe following packages are required but not installed:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nTo install all dependencies, run:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    return True

def main():
    """Launch the application"""
    print("=" * 60)
    print("VISCA Camera Video Tracking Application")
    print("PyQt6 Version")
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("[OK] All dependencies installed\n")
    
    # Import and launch
    try:
        from PyQt6.QtWidgets import QApplication
        from main_app import VideoTrackingApp
        
        print("Launching application...")
        app = QApplication(sys.argv)
        window = VideoTrackingApp()
        window.show()
        
        print("[OK] Application launched successfully!")
        print("\nClose the application window to exit.\n")
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"\n[ERROR] Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
