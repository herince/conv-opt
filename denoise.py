import os
import numbers
import imageio.v2 as iio
import matplotlib.pyplot as plt

test_files_dir = "test-images"
output_dir = "output"

test_files = os.listdir(test_files_dir)

for file_name in test_files:
    file_path = test_files_dir + "/" + file_name
    print("File:", file_path)
    
    img = iio.imread(file_path)

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

    # TODO: Figure out how to denoise an image

    # TODO: MSE

    # TODO: PSNR

    print()
