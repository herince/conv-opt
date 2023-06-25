import os
import numbers
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage

test_files_dir = "test-images/"
output_dir = "output/"

test_files = os.listdir(test_files_dir)

def write_file(path, img):
    print("Output:", path)
    iio.imwrite(path, img)

def generate_diff(original_img, denoised_img, diff_file_path):
    img_diff = denoised_img - original_img
    write_file(diff_file_path, img_diff)

def divergence(m):
    n = len(m)
    return np.ufunc.reduce(np.add, [np.gradient(m[i], axis=i) for i in range(n)])

def normalize(img):
    if img.ndim == 3:
        # TODO
        return img

    else:
        minimum = min(map(min, img))
        maximum = max(map(max, img))

        for i in range(0, len(img)):
            for j in range(0, len(img[i])):
                if (maximum - img[i][j]) > 0:
                    img[i][j] = (img[i][j] - minimum) / ((maximum - minimum))
                else:
                    img[i][j] = 0
    return img

for file_name in test_files:
    file_path = test_files_dir + file_name
    print("File:", file_path)
    
    img = iio.imread(file_path)

    # Apply Gaussian noise
    g_img_out = (skimage.util.random_noise(img, mode="gaussian", mean=0, var=0.00025) * 255).astype(np.uint8) 
    write_file(output_dir + "gaussian-noise/" + file_name, g_img_out)

    # Apply Poisson noise
    p_img_out = (skimage.util.random_noise(img, mode="poisson") * 255).astype(np.uint8)
    write_file(output_dir + "poisson-noise/" + file_name, p_img_out)

    # Apply salt-and-pepper noise
    sp_img_out = (skimage.util.random_noise(img, mode="s&p", amount=0.25) * 255).astype(np.uint8)
    write_file(output_dir + "salt-and-pepper/" + file_name, sp_img_out)

    # Gradient descent
    tau = 0.1
    lambda_ = 50
    denoised_img = g_img_out.astype(float)

    initial_max = max(map(max, denoised_img))

    denoised_img = normalize(denoised_img)

    for _ in range(1, 50):
        gr = np.gradient(denoised_img)
        denoised_img = denoised_img - tau * (denoised_img - g_img_out - lambda_ * divergence(gr))
        denoised_img = normalize(denoised_img)

    denoised_img = (denoised_img * initial_max).astype(np.uint8)
    write_file(output_dir + "denoised-g/" + file_name, denoised_img)

    # L2H1

    # Generate diffs
    generate_diff(denoised_img, img, output_dir + "diff-g/" + file_name)
    # generate_diff(g_img_out, img, output_dir + "diff-p/" + file_name)
    # generate_diff(sp_img_out, img, output_dir + "diff-sp/" + file_name)

    # TODO: MSE

    # TODO: PSNR

    print()
