import os

def make_folder(path):
    """Create a folder if it doesn't exist"""
    os.makedirs(path, exist_ok=True)