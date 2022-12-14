import cv2
import numpy as np

import matplotlib.pyplot as plt
from python_utils import *

metrics = [
    "RMSE",
    "PSNR",
    "SSIM",
]

def main():
    # img1 = cv2.imread('../results/Books/Books_window_1_naive.png')
    # img2 = cv2.imread('../results/Books/Books_window_1_dp.png')
    # img3 = cv2.imread('../results/Books/Books_window_1_sgbm.png')
    # disp_image(img1, img2, img3)

    for metric in metrics:
        fig, ax = plt.subplots(3, 4, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f"{metric.upper()} vs window size for 12 pairs and 3 algorithms")
        bar_plots_metrics(metric, ax)

    fig, ax = plt.subplots(3, 4, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle("Processing Time (s) vs window_size for 12 pairs and 3 algorithms")
    bar_plots_time(ax)

    fig = plt.figure()
    x, y = line_plot_lambda("RMSE")
    print("Optimal lambda for min SSD for Aloe Image: ", x[np.argmin(y[0])])
    print("Optimal lambda for min SSD for Art Image: ", x[np.argmin(y[1])])
    print("Optimal lambda for min SSD for Baby1 Image: ", x[np.argmin(y[2])])

    fig = plt.figure()
    x, y = line_plot_lambda("SSIM")
    print("Optimal lambda for max SSIM for Art Image: ", x[np.argmax(y[0])])
    print("Optimal lambda for max SSIM for Books Image: ", x[np.argmax(y[1])])
    print("Optimal lambda for max SSIM for Dolls Image: ", x[np.argmax(y[2])])
    

    # Load images
    img1 = cv2.imread('../results/Art_window_3_IU.png')
    img2 = cv2.imread('../data/Art/disp1.png')

    diff_image(img1, img2)

    img1 = cv2.imread('../results/Art_window_5_IU.png')
    img2 = cv2.imread('../data/Art/disp1.png')

    diff_image(img1, img2)

    img1 = cv2.imread('../results/Art_window_3_JBMU.png')
    img2 = cv2.imread('../data/Art/disp1.png')

    diff_image(img1, img2)

    plt.show()

if __name__ == "__main__":
    main()

