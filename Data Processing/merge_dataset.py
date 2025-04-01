import os
import pandas as pd

FOLDER = "./DataCSV"

df = pd.DataFrame()
for file in os.listdir(FOLDER):
    if file[-3:] == "csv":
        if df is None or df.empty:
            df = pd.read_csv(f"{FOLDER}/{file}")
        else:
            df = pd.concat([df, pd.read_csv(f"{FOLDER}/{file}")], ignore_index=True)
        
    
df.to_csv("./Data Processing/total.csv", index=False)