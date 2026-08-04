import os

def print_tree(path, prefix="", ignore=None):
    if ignore is None:
        ignore = {
            ".git", "__pycache__", ".venv-hinemo", 
            ".venv", "node_modules", ".DS_Store",
            ".ipynb_checkpoints"
        }
    
    # Get all items, sorted — folders first then files
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return

    # Separate folders and files
    folders = [i for i in items if os.path.isdir(os.path.join(path, i)) and i not in ignore]
    files   = [i for i in items if os.path.isfile(os.path.join(path, i)) and i not in ignore]
    
    all_items = folders + files
    
    for idx, item in enumerate(all_items):
        is_last   = idx == len(all_items) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        full_path = os.path.join(path, item)
        
        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{item}/")
            print_tree(full_path, prefix + extension, ignore)
        else:
            # Show file size for large files
            size = os.path.getsize(full_path)
            if size > 1_000_000:
                size_str = f"  [{size/1_000_000:.1f} MB]"
            elif size > 1_000:
                size_str = f"  [{size/1_000:.1f} KB]"
            else:
                size_str = ""
            print(f"{prefix}{connector}{item}{size_str}")

# Run from project root
ROOT = "/Users/harshaggarwal/Projects_4/hinemo_project"
print(f"{ROOT}/")
print_tree(ROOT)