import sys
import os
from quartodoc import build

# Add root of repo to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

build()

