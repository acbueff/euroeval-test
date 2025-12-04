import os
from pathlib import Path

def get_berzelius_data_path(path_file="documentation/BERZ-PATH.md"):
    """
    Reads the data path from the specified markdown file.
    """
    # Try to locate the file relative to project root
    project_root = Path(__file__).parent.parent
    file_path = project_root / path_file
    
    if not file_path.exists():
        # Fallback if running from a different directory
        file_path = Path(path_file)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find path file at {file_path}")
        
    with open(file_path, 'r') as f:
        content = f.read().strip()
        # Remove any markdown formatting if present (though the example showed just the path)
        return content.split('\n')[0].strip()

def set_hf_cache_env():
    """
    Sets the HF_DATASETS_CACHE and SCANDEVAL_CACHE environment variables 
    to the path found in BERZ-PATH.md.
    """
    try:
        data_path = get_berzelius_data_path()
        os.environ["HF_DATASETS_CACHE"] = data_path
        os.environ["SCANDEVAL_CACHE"] = data_path
        print(f"Set dataset cache paths to: {data_path}")
        return data_path
    except Exception as e:
        print(f"Warning: Could not set cache paths: {e}")
        return None

