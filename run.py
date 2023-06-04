import os
import imageio.v3 as iio
from scipy import ndimage

test_files_dir = "test-images"
output_dir = "output"

test_files = os.listdir(test_files_dir)

for file_name in test_files:
    file_path = test_files_dir + "/" + file_name
    print("File:", file_path)
    
    img = iio.imread(file_path)
    print("Shape:", img.shape)
    
    # todo: solve problem here instead of rotating the image
    out_img = ndimage.rotate(img, 180)
    
    out_file_path = output_dir + "/" + file_name
    print("Output:", out_file_path)
    iio.imwrite(out_file_path, out_img)

    print()
