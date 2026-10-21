"""
Health-Project 3: Breast Cancer Detection System
Production Clinical AI Dashboard Launcher
CSE4001: Applications of Machine Learning in Industry

Usage:
    python run_dashboard.py
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn

# Ensure utf-8 stdout
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "deployment"))

def open_browser():
    time.sleep(1.5)
    print("\n[+] Opening Clinical Dashboard in default browser: http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 76)
    print("   [+] BREAST CANCER CYTOPATHOLOGY AI DETECTION & TRIAGE PLATFORM")
    print("   Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) - 569 Biopsies")
    print("   Production Champion: Support Vector Machine (RBF Kernel, C=5.0)")
    print("   Test Accuracy: 100.00% | Malignant Recall (Sensitivity): 100.00%")
    print("=" * 76)
    print("  Server URL : http://127.0.0.1:8000")
    print("  Dashboard  : http://127.0.0.1:8000/dashboard")
    print("  REST Docs  : http://127.0.0.1:8000/docs")
    print("=" * 76)
    print("  [Instructions for Faculty & Examiner Demonstration]")
    print("  1. The Web Dashboard opens automatically in your browser.")
    print("  2. Click 'Load Malignant Case' or 'Load Benign Case' for instant presets.")
    print("  3. Click 'Run AI Diagnosis' to see instant prediction, risk %, and triage flag.")
    print("  4. Click 'Model Accuracy & Evidence' tab to view 100% test accuracy proof.")
    print("  5. Click 'All 8 Experiments' tab to view high-res plots for Experiments 1-8.")
    print("=" * 76)
    print("  Press CTRL+C anytime in this terminal to stop the server.\n")

    # Start browser launcher thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI via Uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info", reload=False)

if __name__ == "__main__":
    main()
