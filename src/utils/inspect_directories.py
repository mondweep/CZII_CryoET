from pathlib import Path
import os
from dotenv import load_dotenv

def inspect_directory(path, level=0, max_depth=3):
    """
    Recursively inspect directory structure up to max_depth
    Returns a list of files found with .zarr extension
    """
    zarr_files = []
    path = Path(path)
    indent = "  " * level
    
    try:
        # List contents
        contents = list(path.iterdir())
        print(f"{indent}📁 {path.name}/")
        
        # Stop if we've reached max depth
        if level >= max_depth:
            if any(item.name.endswith('.zarr') for item in contents):
                print(f"{indent}  [...contains .zarr files...]")
            return zarr_files
        
        # Process contents
        for item in contents:
            if item.is_dir():
                if item.name.endswith('.zarr'):
                    print(f"{indent}  📦 {item.name}")
                    zarr_files.append(str(item))
                else:
                    zarr_files.extend(inspect_directory(item, level + 1, max_depth))
            elif item.name.endswith('.csv'):
                print(f"{indent}  📄 {item.name}")
                
        return zarr_files
                
    except PermissionError:
        print(f"{indent}❌ Permission denied: {path}")
        return zarr_files
    except Exception as e:
        print(f"{indent}❌ Error reading {path}: {str(e)}")
        return zarr_files

def main():
    # Load environment variables
    load_dotenv()
    
    # Paths to inspect
    paths = {
        "DATASET_PATH": os.getenv('DATASET_PATH'),
        "ANNOTATED_DATASET_PATH": os.getenv('ANNOTATED_DATASET_PATH'),
        "TEST_DATASET_PATH": os.getenv('TEST_DATASET_PATH')
    }
    
    # Inspect each path
    for name, path in paths.items():
        if not path:
            print(f"\n❌ {name} is not set!")
            continue
            
        if not os.path.exists(path):
            print(f"\n❌ {name} does not exist: {path}")
            continue
            
        print(f"\n🔍 Inspecting {name}:")
        print(f"Path: {path}")
        zarr_files = inspect_directory(path)
        
        if zarr_files:
            print(f"\nFound {len(zarr_files)} .zarr files:")
            for zarr_file in zarr_files[:5]:  # Show first 5
                print(f"  - {zarr_file}")
            if len(zarr_files) > 5:
                print(f"  ... and {len(zarr_files)-5} more")

if __name__ == "__main__":
    main() 