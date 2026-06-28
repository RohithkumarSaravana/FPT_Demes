"""
download_datasets.py
Downloads all 5 datasets into the datasets/ folder.
Run once from inside the met2/ folder:
    python codes/download_datasets.py
"""
import urllib.request, os

os.makedirs("datasets", exist_ok=True)

datasets = {
    "australian.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat",
    "pima.data":       "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
    "heart.data":      "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
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
        print(f"  ERROR downloading {filename}: {e}")

print("\nDone. Verify with:")
print("  python -c \"import pandas as pd; print(pd.read_csv('datasets/australian.data', sep=' ', header=None).shape)\"")
print("  python -c \"import pandas as pd; print(pd.read_csv('datasets/pima.data', sep=',', header=None).shape)\"")
print("  python -c \"import pandas as pd; print(pd.read_csv('datasets/heart.data', sep=' ', header=None).shape)\"")
