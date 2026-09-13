"""Single-process entrypoint for the hosted interview demonstration."""
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ['MSME_EMBEDDED_API'] = '1'
runpy.run_path(str(root / 'dashboard.py'), run_name='__main__')
