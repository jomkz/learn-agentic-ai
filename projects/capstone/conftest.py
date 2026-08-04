import sys
from pathlib import Path

# Add capstone dir to sys.path
sys.path.insert(0, str(Path(__file__).parent))
# Add evals dir for ragas_harness
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
# Add all phase directories for cross-phase imports
for phase_dir in sorted(Path(__file__).parent.parent.glob("phase*/")):
    sys.path.insert(0, str(phase_dir))
