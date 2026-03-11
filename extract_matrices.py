import os
import cv2
import numpy as np
import pandas as pd
import math
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

def resize_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, size)
    return resized_image

def calculate_metrics_grayscale(image1, image2):
    gray_image1 = rgb2gray(image1 / 255.0)
    gray_image2 = rgb2gray(image2 / 255.0)

    mse = np.mean((gray_image1 - gray_image2) ** 2)

    if mse == 0:
        psnr_value = 100
    else:
        max_pixel = 1.0
        psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))

    ssim_value = structural_similarity(gray_image1, gray_image2, multichannel=False, data_range=1.0)

    return round(psnr_value, 3), round(mse, 3), round(math.sqrt(mse), 3), round(ssim_value, 3)

def main(fake_folder, real_folder, output_csv):
    results = []

    fake_images = sorted(os.listdir(fake_folder))
    real_images = sorted(os.listdir(real_folder))

    for fake_image_name in fake_images:
        if '3T' in fake_image_name:
            real_image_name = fake_image_name.replace('3T', '7T')
        else:
            continue

        if real_image_name in real_images:
            fake_image_path = os.path.join(fake_folder, fake_image_name)
            real_image_path = os.path.join(real_folder, real_image_name)

            fake_image = resize_image(fake_image_path)
            real_image = resize_image(real_image_path)

            print(f"Processing: {fake_image_name} and {real_image_name}")

            psnr_value, mse_value, rmse_value, ssim_value = calculate_metrics_grayscale(fake_image, real_image)

            results.append({
                'Fake Image': fake_image_name,
                'Real Image': real_image_name,
                'PSNR': psnr_value,
                'MSE': mse_value,
                'RMSE': rmse_value,
                'SSIM': ssim_value
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    fake_folder = 'fake_dataset_7T_780'
    real_folder = '7T_testing_dataset'
    output_csv = 'image_comparison_results_780.csv'

    main(fake_folder, real_folder, output_csv)