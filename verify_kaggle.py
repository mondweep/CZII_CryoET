import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

def verify_kaggle_connection():
    try:
        # Load environment variables
        load_dotenv()
        
        # Ensure Kaggle credentials are set in environment
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
        
        print(f"Using Kaggle username: {os.getenv('KAGGLE_USERNAME')}")
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("Successfully authenticated with Kaggle!")
        print("\nAttempting to access competition...")
        
        # Competition ID from the URL
        COMPETITION_ID = 'czii-cryo-et-object-identification'
        
        try:
            # List competition files
            print("\nCompetition files available:")
            files = api.competition_list_files(COMPETITION_ID)
            for file in files:
                print(f"- {file.name} ({file.size} bytes)")
            
            print("Successfully verified competition access!")
            
        except Exception as e:
            print(f"\nError accessing competition: {e}")
            print("\nTroubleshooting steps:")
            print("1. Try downloading directly from the competition page")
            print("2. Verify your Kaggle account email is verified")
            print("3. Try logging out and back in to Kaggle")
            
            # Print current environment setup
            print("\nCurrent environment setup:")
            print(f"KAGGLE_USERNAME set: {'Yes' if os.getenv('KAGGLE_USERNAME') else 'No'}")
            print(f"KAGGLE_KEY set: {'Yes' if os.getenv('KAGGLE_KEY') else 'No'}")
            
    except Exception as e:
        print(f"Authentication error: {e}")

if __name__ == "__main__":
    verify_kaggle_connection() 