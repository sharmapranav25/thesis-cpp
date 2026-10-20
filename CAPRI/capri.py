"""Single entry point for the maintained CAPRI controller workflow."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from capri_controller.cli import main
if __name__=='__main__':main()
