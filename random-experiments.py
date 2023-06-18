import os
import numbers
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import skimage

# def dilate(img, str):
# def erode(img, str):
# def apply_closing(img, str):
# def apply_opening(img, str):

test_files_dir = "test-images"
output_dir = "output"

test_files = os.listdir(test_files_dir)

def segment(n, th):
    if isinstance(n, numbers.Number):
        if n <= th:
            return 0
        return 255

    if n.all() <= th:
        return np.full(len(n), 0)
    return np.full(len(n), 255)

for file_name in test_files:
    file_path = test_files_dir + "/" + file_name
    print("File:", file_path)
    
    img = iio.imread(file_path)

    img_out = img
    # for row, _ in enumerate(img):
    #     for column, _ in enumerate(img[row]):
    #         img_out[row][column] = segment(img[row][column], 125)

    # Apply Gaussian noise
    img_out = (skimage.util.random_noise(img, mode="gaussian", mean=0.25, var=0.05) * 255).astype(np.uint8)
    out_file_path = output_dir + "/gaussian-noise/" + file_name
    print("Output:", out_file_path)
    iio.imwrite(out_file_path, img_out)

    # Apply Poisson noise
    img_out = (skimage.util.random_noise(img, mode="poisson") * 255).astype(np.uint8)
    out_file_path = output_dir + "/poisson-noise/" + file_name
    print("Output:", out_file_path)
    iio.imwrite(out_file_path, img_out)

    # Apply salt-and-pepper noise
    img_out = (skimage.util.random_noise(img, mode="s&p", amount=0.25) * 255).astype(np.uint8)
    out_file_path = output_dir + "/salt-and-pepper/" + file_name
    print("Output:", out_file_path)
    iio.imwrite(out_file_path, img_out)

    print()
