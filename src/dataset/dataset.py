# download_pascal_voc.py

from kaggle.api.kaggle_api_extended import KaggleApi
import os

def download_pascal_voc(dataset="gopalbhattrai/pascal-voc-2012-dataset",  
                       download_path="data/pascal_voc_2012",
                       unzip=True):
    """
    Downloads the VOC 2012 dataset from Kaggle to the specified path.
    """
    api = KaggleApi()
    api.authenticate()
    
    # Create target directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download all files
    api.dataset_download_files(dataset, path=download_path, unzip=unzip)
    print(f"Downloaded dataset to: {download_path}")

if __name__ == "__main__":
    download_pascal_voc()
