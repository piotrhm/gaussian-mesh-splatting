import os
from PIL import Image

# Set the path to your folder and the image filename
folder_path = 'data/person_1_single/train'
folder_path_single = 'data/person_1_single/train_s'
image_filename = 'f_0000.png'

# Full path to the original image
image_path = os.path.join(folder_path, image_filename)

# Ensure the image exists
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"The image {image_filename} does not exist in the specified folder.")

# Ensure the target directory exists
if not os.path.exists(folder_path_single):
    os.makedirs(folder_path_single)

# Open the original image
original_image = Image.open(image_path)
# Create 100 copies
for i in range(100):
    # Create a new filename for each copy following the pattern f_0000, f_0001, etc.
    new_filename = f"f_{i:04d}{os.path.splitext(image_filename)[1]}"
    new_image_path = os.path.join(folder_path_single, new_filename)
    # Save the image as a new file
    original_image.save(new_image_path)

print("100 copies have been created.")