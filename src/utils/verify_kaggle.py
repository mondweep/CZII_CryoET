import os
from kaggle.api.kaggle_api_extended import KaggleApi

def verify_kaggle_credentials():
    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()
        print("Kaggle authentication successful!")
        
        # List available competitions to verify connection
        competitions = api.competitions_list()
        print("\nAvailable competitions:")
        for comp in competitions[:3]:  # Show first 3 competitions
            print(f"- {comp.title}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a Kaggle account")
        print("2. Generated an API token from https://www.kaggle.com/account")
        print("3. Placed kaggle.json in ~/.kaggle/ directory")

if __name__ == "__main__":
    verify_kaggle_credentials() 