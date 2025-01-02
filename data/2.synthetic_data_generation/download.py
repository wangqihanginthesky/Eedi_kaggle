import kagglehub
import shutil
# Specify the download directory
for i in range(3):
    round = i+1
    custom_path = f"synthetic-round{round}-render.csv"
    # Download dataset to the specified directory
    path = kagglehub.dataset_download("wangqihanginthesky/eedi-2nd-place-synthetic-data", path=custom_path)
    shutil.move(path,f"./generation{round}")