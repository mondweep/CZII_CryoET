import os
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import subprocess

def download_competition_files():
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        COMPETITION_ID = 'czii-cryo-et-object-identification'
        
        print(f"\nDownloading competition files to Kaggle workspace...")
        process = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', COMPETITION_ID],
            timeout=180,  # 3 minute timeout
            capture_output=True
        )
        
        if process.returncode == 0:
            print("\nFiles downloaded successfully!")
            subprocess.run(['ls', '-l', '*.zip'], shell=True, timeout=10)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_competition_files() 