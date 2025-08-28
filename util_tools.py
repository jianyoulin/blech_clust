import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

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


def save_subfolders_to_file(parent_folder, output_file):
    """
    Retrieves the full paths of all subfolders in a parent directory
    and saves them to a text file, one path per line.
    """
    try:
        # Check if the parent folder exists
        if not os.path.isdir(parent_folder):
            print(f"Error: The folder '{parent_folder}' does not exist.")
            return

        # Open the output file in write mode ('w')
        file_saved = os.path.join(parent_folder, output_file)
        with open(file_saved, 'w') as f:
            # Use os.scandir to iterate through the contents of the parent folder
            with os.scandir(parent_folder) as entries:
                for entry in entries:
                    # Check if the entry is a directory
                    if entry.is_dir():
                        # Write the full path of the directory to the file
                        f.write(entry.path + '\n')
        
        print(f"Successfully saved all subfolder paths to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

def show_png(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
    else:
        # Open the image using Pillow
        try:
            img = Image.open(image_path)
            # Display the image using Matplotlib
            plt.imshow(img)
            plt.title(f"Displaying {image_path}")
            plt.axis('off')  # Hide the axes for a cleaner display
            plt.show()

            print(f"Successfully displayed '{image_path}'.")

        except IOError:
            print(f"Error: Could not open or read the image file '{image_path}'.")

            
def padding_list(lists, padding_value=999):
    """Pad lists with 999 to make them the same length.

    This function takes a list of lists and pads each inner list with NaN values
    so that all inner lists have the same length. The padding is done at the end
    of each list.

    Args:
        lists (list of list): A list containing multiple lists of varying lengths.

    Returns:
        list of list: A new list where all inner lists are padded with NaNs to match
                      the length of the longest inner list.
    """
    max_length = max(len(lst) for lst in lists)
    padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in lists]
    return padded_lists            