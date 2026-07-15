import sys
from pathlib import Path

# Add this phase directory to sys.path so tests can import phase modules directly.
# Necessary because directory names with hyphens can't be Python package names.
sys.path.insert(0, str(Path(__file__).parent))
