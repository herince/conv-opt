import os
import imageio.v2 as iio
import numpy as np
import skimage
import math
import matplotlib.pyplot as plt

def write_file(path, img):
    print("Output:", path)
    iio.imwrite(path, img)

def generate_diff(original_img, denoised_img, diff_file_path):
    img_diff = original_img
    for i in range(0, len(original_img)):
        for j in range(0, len(original_img[i])):
            img_diff[i][j] = abs(original_img[i][j].astype(np.int16) - denoised_img[i][j].astype(np.int16))
    img_diff = img_diff.astype(np.uint8)
    
    write_file(diff_file_path, img_diff)

    return img_diff

def divergence(m):
    n = len(m)
    return np.ufunc.reduce(np.add, [np.gradient(m[i], axis=i) for i in range(n)])

def inverse_normalize(img, diff_list, min_list):
    for i in range(0, len(diff_list)):
        img = img * diff_list[len(diff_list) - i - 1] + min_list[len(min_list) - i - 1]
    return img

def gradient_descent(path, img, expected):
    tau = 0.05
    lambda_ = 1
    steps = 5

    diff_list = []
    min_list = []

    img_f = img.astype(float)
    [normalized_img, diff, minimum] = normalize(img_f)
    diff_list.append(diff)
    min_list.append(minimum)

    denoised_img = normalized_img

    for i in range(0, steps):
        gr = np.gradient(denoised_img)
        denoised_img = denoised_img - tau * (denoised_img - normalized_img - lambda_ * divergence(gr))

        [denoised_img, diff, minimum] = normalize(denoised_img)
        diff_list.append(diff)
        min_list.append(minimum)

    denoised_img = inverse_normalize(denoised_img, diff_list, min_list)
    denoised_img = denoised_img.round().astype(np.uint8)

    print(f"MAE: {mae(expected, denoised_img)}")
    print(f"PSNR: {psnr(expected, denoised_img)}")

    write_file(path, denoised_img)

    return denoised_img

def normalize(img):
    if img.ndim == 3:
        return normalize_rgb(img)

    minimum = min(map(min, img))
    maximum = max(map(max, img))
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if (maximum - img[i][j]) > 0:
                img[i][j] = (img[i][j] - minimum) / (maximum - minimum)
            else:
                img[i][j] = 0

    return img, maximum - minimum, minimum

def mae(img, denoised_img):
    return np.mean(img - denoised_img)

def mse(img, denoised_img):
    return np.mean((img - denoised_img) ** 2)

def psnr(img, denoised_img):
    mse_value = mse(img, denoised_img)
    if mse_value == 0:
        return 100
    max_value = 255.0
    psnr = 20 * math.log10(max_value / math.sqrt(mse_value))
    return psnr

if __name__ == "__main__":
    test_files_dir = "test-images/"
    output_dir = "output/"

    test_files = os.listdir(test_files_dir)

    for file_name in test_files:
        file_path = test_files_dir + file_name
        print("File:", file_path)
        
        img = iio.imread(file_path)
        if img.ndim == 3:
            print("Skip RGB image\n")
            continue

        # Apply Gaussian noise
        g_img_out = (skimage.util.random_noise(img, mode="gaussian", var=0.005) * 255).astype(np.uint8) 
        write_file(f"{output_dir}gaussian-noise/{file_name}", g_img_out)

        # Apply Poisson noise
        p_img_out = (skimage.util.random_noise(img, mode="poisson") * 255).astype(np.uint8)
        write_file(f"{output_dir}poisson-noise/{file_name}", p_img_out)

        # Apply salt-and-pepper noise
        sp_img_out = (skimage.util.random_noise(img, mode="s&p", amount=0.0025) * 255).astype(np.uint8)
        write_file(f"{output_dir}salt-and-pepper/{file_name}", sp_img_out)

        # Gradient descent
        g_img_denoised = gradient_descent(f"{output_dir}denoised-g/{file_name}", g_img_out, img)
        p_img_denoised = gradient_descent(f"{output_dir}denoised-p/{file_name}", p_img_out, img)
        sp_img_denoised = gradient_descent(f"{output_dir}denoised-sp/{file_name}", sp_img_out, img)

        # Generate diffs
        g_diff = generate_diff(g_img_denoised, img, f"{output_dir}diff-g/{file_name}")
        p_diff = generate_diff(p_img_denoised, img, f"{output_dir}diff-p/{file_name}")
        sp_diff = generate_diff(sp_img_denoised, img, f"{output_dir}diff-sp/{file_name}")

        print()
