"""
TechText AI - Environment & Dependency Verification Script
Run: python check_setup.py
"""

import sys
import os
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def check_python_version():
    print(f"Python Version: {sys.version}")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 9):
        print(f" [PASS] Python {major}.{minor} meets requirement (>= 3.9)")
        return True
    else:
        print(f" [FAIL] Python {major}.{minor} is too old. Please use Python 3.9+")
        return False


def check_dependencies():
    packages = [
        ("torch", "PyTorch (Deep Learning)"),
        ("numpy", "NumPy (Numerical Computing)"),
        ("pandas", "Pandas (DataFrames)"),
        ("requests", "Requests (HTTP Client)"),
        ("bs4", "BeautifulSoup4 (HTML Parsing)"),
        ("nltk", "NLTK (Natural Language Toolkit)"),
        ("fastapi", "FastAPI (Backend Web Framework)"),
        ("uvicorn", "Uvicorn (ASGI Server)"),
        ("pydantic", "Pydantic (Data Validation)"),
        ("tiktoken", "TikToken (BPE Tokenizer)"),
        ("transformers", "Transformers (Hugging Face)"),
        ("sklearn", "Scikit-Learn (Machine Learning)"),
    ]

    all_passed = True
    print("\nChecking Packages:")
    for pkg_name, label in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [PASS] {label:35} v{ver}")
        except ImportError as e:
            print(f"  [FAIL] {label:35} NOT FOUND ({e})")
            all_passed = False
    return all_passed


def check_hardware():
    print("\nHardware Acceleration:")
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            print(f"  [INFO] CUDA is AVAILABLE: {device_name}")
            print(f"  [INFO] Device count: {torch.cuda.device_count()}")
        else:
            print("  [INFO] CUDA not detected. Training will run on CPU (fully supported).")
    except Exception as e:
        print(f"  [WARN] Could not inspect PyTorch device: {e}")


def check_directories():
    required_dirs = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "src/scraper",
        "src/nlp",
        "src/model",
        "src/backend",
        "frontend",
        "tests",
    ]

    print("\nChecking Project Directories:")
    all_exist = True
    root = Path(__file__).resolve().parent

    for d in required_dirs:
        dir_path = root / d
        if dir_path.is_dir():
            print(f"  [PASS] Directory exists: {d}")
        else:
            print(f"  [WARN] Missing directory: {d} (creating now...)")
            dir_path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    return all_exist


def main():
    print_header("TechText AI: Environment Verification")

    py_ok = check_python_version()
    deps_ok = check_dependencies()
    check_hardware()
    dirs_ok = check_directories()

    print_header("Verification Summary")
    if py_ok and deps_ok and dirs_ok:
        print(" ALL CHECKS PASSED! TechText AI environment is ready.")
        return 0
    else:
        print(" SOME CHECKS FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
