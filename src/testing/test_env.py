from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Print all relevant environment variables
print("Environment Variables:")
print(f"TEST_DATASET_PATH: {os.getenv('TEST_DATASET_PATH')}")
print(f"DATASET_PATH: {os.getenv('DATASET_PATH')}")
print(f"ANNOTATED_DATASET_PATH: {os.getenv('ANNOTATED_DATASET_PATH')}")

# Verify paths exist
for path_name, path in [
    ("TEST_DATASET_PATH", os.getenv('TEST_DATASET_PATH')),
    ("DATASET_PATH", os.getenv('DATASET_PATH')),
    ("ANNOTATED_DATASET_PATH", os.getenv('ANNOTATED_DATASET_PATH'))
]:
    if path:
        exists = os.path.exists(path)
        print(f"\n{path_name}:")
        print(f"Path: {path}")
        print(f"Exists: {exists}")
        if exists:
            print(f"Contents: {os.listdir(path)[:5]}")  # Show first 5 items
    else:
        print(f"\n{path_name} is not set") 