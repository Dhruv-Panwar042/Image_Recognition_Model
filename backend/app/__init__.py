import sys
from pathlib import Path

# Ensure backend directory is at the front of sys.path for clean package imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

"""Intelligent Image Analysis System Backend Application Package."""
