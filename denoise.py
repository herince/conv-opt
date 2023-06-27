import os
import imageio.v2 as iio
import numpy as np
import skimage
import math

def write_file(path, img):
    print("Output:", path)
    iio.imwrite(path, img)

def generate_diff(original_img, denoised_img, diff_file_path):
    img_diff = original_img - denoised_img
    write_file(diff_file_path, img_diff)

def divergence(m):
    n = len(m)
    return np.ufunc.reduce(np.add, [np.gradient(m[i], axis=i) for i in range(n)])

def gradient_descent(path, img):
    tau = 0.1
    lambda_ = 2
    img_f = img.astype(float)
    normalized_img_f = normalize(img_f)
    denoised_img = normalized_img_f

    for _ in range(1, 50):
        # Тhe gradient and maybe the divergence (?) need to be calculated differently for RGB images
        gr = np.gradient(denoised_img)
        denoised_img = denoised_img - tau * (denoised_img - normalized_img_f - lambda_ * divergence(gr))
        denoised_img = normalize(denoised_img)

    denoised_img = (denoised_img * 255).round().astype(np.uint8)
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
                img[i][j] = (img[i][j] - minimum) / ((maximum - minimum))
            else:
                img[i][j] = 0
    return img

def normalize_rgb(img):
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            s = math.sqrt(sum(img[i][j] ** 2))
            if s > 0:
                img[i][j] = [img[i][j][0] / s, img[i][j][1] / s, img[i][j][2] / s]
            else:
                img[i][j] = [0, 0, 0]
    return img

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

        # Apply Gaussian noise
        g_img_out = (skimage.util.random_noise(img, mode="gaussian", var=0.00025) * 255).astype(np.uint8) 
        write_file(f"{output_dir}gaussian-noise/{file_name}", g_img_out)

        # Apply Poisson noise
        p_img_out = (skimage.util.random_noise(img, mode="poisson") * 255).astype(np.uint8)
        write_file(f"{output_dir}poisson-noise/{file_name}", p_img_out)

        # Apply salt-and-pepper noise
        sp_img_out = (skimage.util.random_noise(img, mode="s&p", amount=0.00025) * 255).astype(np.uint8)
        write_file(f"{output_dir}salt-and-pepper/{file_name}", sp_img_out)

        # Gradient descent
        g_img_denoised = gradient_descent(f"{output_dir}denoised-g/{file_name}", g_img_out)
        p_img_denoised = gradient_descent(f"{output_dir}denoised-p/{file_name}", p_img_out)
        sp_img_denoised = gradient_descent(f"{output_dir}denoised-sp/{file_name}", sp_img_out)

        # L2H1

        # Generate diffs
        generate_diff(g_img_denoised, img, f"{output_dir}diff-g/{file_name}")
        generate_diff(p_img_denoised, img, f"{output_dir}diff-p/{file_name}")
        generate_diff(sp_img_denoised, img, f"{output_dir}diff-sp/{file_name}")

        # MAE
        print(f"MAE: {mae(img, g_img_denoised)}")
        print(f"MAE: {mae(img, p_img_denoised)}")
        print(f"MAE: {mae(img, sp_img_denoised)}")

        # MSE
        print(f"MSE: {mse(img, g_img_denoised)}")
        print(f"MSE: {mse(img, p_img_denoised)}")
        print(f"MSE: {mse(img, sp_img_denoised)}")

        # PSNR
        print(f"PSNR: {psnr(img, g_img_denoised)}")
        print(f"PSNR: {psnr(img, p_img_denoised)}")
        print(f"PSNR: {psnr(img, sp_img_denoised)}")

        print()
