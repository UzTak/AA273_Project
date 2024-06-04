import os
import imageio

# Relative path to the folder containing images
relative_folder_path = 'simulation_imgs'
folder_path = os.path.abspath(relative_folder_path)

# Generate list of filenames from 0.png to 70.png
filenames = [f"{i}.png" for i in range(71)]
print(filenames)
# Create full file paths and check if they exist
files = [os.path.join(folder_path, filename) for filename in filenames if os.path.exists(os.path.join(folder_path, filename))]

print(files)
# Read images
images = [imageio.imread(file) for file in files]

output_file = 'output.mp4'
imageio.mimwrite(output_file, images, fps=6, codec='libx264')

print(f"Video saved as {output_file}")
