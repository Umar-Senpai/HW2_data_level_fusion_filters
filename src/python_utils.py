import cv2
import numpy as np

from skimage.metrics import structural_similarity as skt_ssim 
import matplotlib.pyplot as plt
import re


def read_image(path: str):
    return cv2.imread(path)

def ssim(img1, img2) -> float:
    """
    Structural Simularity Index
    """
    return skt_ssim(img1, img2, channel_axis=2)

def ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

def ncc(img1, img2) -> float:
    """
    Normalized Cross Corelation
    """
    return cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)

def numbers_from_file(file):
        """From a list of integers in a file, creates a list of tuples"""
        with open(file, 'r') as f:
            return([float(x) for x in re.findall(r'[\d]*[.]?[\d]+', f.read())])

metrics = [
    "RMSE",
    "PSNR",
    "SSIM",
]

dataset = ["Aloe", "Art", "Baby1", "Books", "Bowling1", "Dolls", "Flowerpots", "Laundry", "Moebius", "Monopoly", "Reindeer", "Wood1"]
algorithms = ["JBU", "JBMU", "IU"]
window_size = [3, 5, 7]

data_dir = "../results/"

def bar_plots_metrics(metric, ax):
    output_dict = {}
    index = metrics.index(metric)
    index_arr = [0 + index, 3 + index, 6 + index]
    for name in dataset:
        output_dict[name] = {}
        for size in window_size:
            file_name = "{}_window_{}_errors.txt".format(name, size)
            time_arr = numbers_from_file(data_dir + file_name)
            output_dict[name][size] = [time_arr[i] for i in index_arr]

    for id, name in enumerate(dataset):
        first_index = int(id/4)

        x1 = window_size
        y1 = [output_dict[name][3][0], output_dict[name][5][0], output_dict[name][7][0]]

        x2 = window_size
        y2 = [output_dict[name][3][1], output_dict[name][5][1], output_dict[name][7][1]]

        x3 = window_size
        y3 = [output_dict[name][3][2], output_dict[name][5][2], output_dict[name][7][2]]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 4].bar(x1-width, y1, label="JBU", color='b', width=0.4)
        ax[first_index][id % 4].bar(x2, y2, label="JBMU", color='g', width=0.4)
        ax[first_index][id % 4].bar(x3+width, y3, label="IU", color='r', width=0.4)
        ax[first_index][id % 4].plot()

        ax[first_index][id % 4].set_xlabel("window size")
        ax[first_index][id % 4].set_ylabel(metric.upper())
        ax[first_index][id % 4].set_title("{} Image Middlebury".format(name))
        ax[first_index][id % 4].legend()

def bar_plots_time(ax):
    processing_time_dict = {}
    for name in dataset:
        processing_time_dict[name] = {}
        for size in window_size:
            processing_time_dict[name][size] = {}
            file_name = "{}_window_{}_processing_time.txt".format(name, size)
            time_arr = numbers_from_file(data_dir + file_name)
            processing_time_dict[name][size] = time_arr

    for id, name in enumerate(dataset):
        first_index = int(id/4)

        x1 = window_size
        y1 = [processing_time_dict[name][3][0], processing_time_dict[name][5][0], processing_time_dict[name][7][0]]

        x2 = window_size
        y2 = [processing_time_dict[name][3][1], processing_time_dict[name][5][1], processing_time_dict[name][7][1]]

        x3 = window_size
        y3 = [processing_time_dict[name][3][2], processing_time_dict[name][5][2], processing_time_dict[name][7][2]]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 4].bar(x1-width, y1, label="JBU", color='b', width=0.4)
        ax[first_index][id % 4].bar(x2, y2, label="JBMU", color='g', width=0.4)
        ax[first_index][id % 4].bar(x3+width, y3, label="IU * 100", color='r', width=0.4)
        ax[first_index][id % 4].plot()

        ax[first_index][id % 4].set_xlabel("window size")
        ax[first_index][id % 4].set_ylabel("Time (seconds)")
        ax[first_index][id % 4].set_title("{} Image Middlebury".format(name))
        ax[first_index][id % 4].legend()

def line_plot_lambda(metric):
    lambda_val = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    dataset = ["Aloe", "Art", "Baby1"]
    index = metrics.index(metric)
    dp_lambda_dict = {}
    for data in dataset:
        dp_lambda_dict[data] = []
        for val in lambda_val:
            file_name = f"{data}_window_7_sigma_{val:.6f}_errors.txt"
            time_arr = numbers_from_file(data_dir + "sigma/" + file_name)
            dp_lambda_dict[data].append(time_arr[index])

    x  = lambda_val
    y1 = dp_lambda_dict["Aloe"]
    y2 = dp_lambda_dict["Art"]
    y3 = dp_lambda_dict["Baby1"]
    plt.plot(x, y1, label="Aloe Image")
    plt.plot(x, y2, label="Art Image")
    plt.plot(x, y3, label="Baby1 Image")
    plt.plot()

    plt.xlabel("Lambda")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Lambda for DP Algorithm")
    plt.legend(loc="center right")

    return x, [y1, y2, y3]

def diff_image(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = skt_ssim(img1_gray, img2_gray, full=True)

    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    filled_after = img2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].set_title("Upsampled Disparity Image")
    ax[1].imshow(img2)
    ax[1].set_title("Ground Truth Disparity Image")
    ax[2].imshow(filled_after)
    ax[2].set_title("Diff Image (Green shows missing parts in Image 2 w.r.t Image 1)")

def disp_image(img1, img2, img3):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].set_title("Naive Disparity")
    ax[1].imshow(img2)
    ax[1].set_title("Dynamic Programming Disparity")
    ax[2].imshow(img3)
    ax[2].set_title("SGBM Disparity")