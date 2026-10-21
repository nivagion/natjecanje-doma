import argparse
from pathlib import Path
import pandas as pd
from src.reporting import write_final

ap=argparse.ArgumentParser();ap.add_argument("predictions");args=ap.parse_args()
p=Path(args.predictions);print(write_final(p.parent,pd.read_csv(p)))
