import os
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import subprocess

def download_competition_files():
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Competition ID
        COMPETITION_ID = 'czii-cryo-et-object-identification'
        
        try:
            # Download competition files to Kaggle workspace
            print(f"\nDownloading competition files to Kaggle workspace...")
            
            # Use Kaggle's built-in competition download command
            subprocess.run(['kaggle', 'competitions', 'download', '-c', COMPETITION_ID])
            
            print("\nFiles downloaded successfully!")
            
            # List downloaded files in Kaggle workspace
            print("\nDownloaded files in workspace:")
            subprocess.run(['ls', '-l', '*.zip'], shell=True)
            
        except Exception as e:
            print(f"\nError downloading competition files: {e}")
            print("\nTroubleshooting steps:")
            print("1. Make sure you're running this in a Kaggle notebook")
            print("2. Verify you've accepted the competition rules")
            
    except Exception as e:
        print(f"Authentication error: {e}")

if __name__ == "__main__":
    download_competition_files() 