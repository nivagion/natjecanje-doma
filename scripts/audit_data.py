import argparse
import json
import pandas as pd
from src.data import audit

ap=argparse.ArgumentParser();ap.add_argument("files",nargs="+",default=[]);args=ap.parse_args()
for file in args.files:
    df=pd.read_csv(file,low_memory=False)
    print(file,json.dumps(audit(df,labelled="target_index" in df.columns),indent=2))
