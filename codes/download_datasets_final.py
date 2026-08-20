"""
download_datasets_final.py
Downloads all 5 datasets into the datasets/ folder.
Run once from inside the project root folder:
    python codes/download_datasets_final.py
"""
import urllib.request, os, pandas as pd

os.makedirs("datasets", exist_ok=True)

datasets = {
    # Australian credit approval dataset (with header: d0..d13, output)
    "australian.dat": "https://raw.githubusercontent.com/bdsul/grape/main/datasets/australian.dat",
    # Cleveland heart disease dataset (processed.cleveland.data)
    "processed.cleveland.data": "https://raw.githubusercontent.com/bdsul/grape/main/datasets/processed.cleveland.data",
    # Pima from reliable mirror
    "pima.data": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
}

for filename, url in datasets.items():
    path = f"datasets/{filename}"
    if os.path.exists(path):
        print(f"  Already exists: {path} — skipping")
        continue
    print(f"Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path}")
    except Exception as e:
        print(f"  ERROR: {e}")

# Verify
print("\nVerification:")
aus = pd.read_csv("datasets/australian.dat", sep=" ")
print(f"  australian.dat     : {aus.shape} — cols: {list(aus.columns[:4])}...")
print(f"    output values    : {sorted(aus['output'].unique())}")
print(f"    d3 unique        : {sorted(aus['d3'].unique())} (rows with d3=3 will be removed)")

clev = pd.read_csv("datasets/processed.cleveland.data", sep=",")
print(f"  processed.cleveland: {clev.shape} — cols: {list(clev.columns[:4])}...")
print(f"    class values     : {sorted(clev['class'].unique())} (binarised: 0=no disease, 1=disease)")
print(f"    ca unique        : {sorted(clev['ca'].unique())} (? rows removed)")
print(f"    thal unique      : {sorted(clev['thal'].unique())} (? rows removed)")

pima = pd.read_csv("datasets/pima.data", sep=",", header=None)
print(f"  pima.data          : {pima.shape}")
print("\nAll datasets ready.")
