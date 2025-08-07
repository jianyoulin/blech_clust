import os
import shutil

def recreate_folder(folder_path):
    """
    Create a folder at the given path.
    If the folder exists, delete it and create a new one.
    
    Parameters:
    - folder_path (str): Path to the folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes folder and contents
    os.makedirs(folder_path)        # Creates the folder