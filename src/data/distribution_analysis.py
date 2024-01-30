from torchdata.datapipes.iter import IterableWrapper
from src.data.ingestion import combine_and_resize_bands
import os
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

with open('../../aws_credentials.yaml', 'r') as file:
    credentials = yaml.safe_load(file)
os.environ["AWS_ACCESS_KEY_ID"] = credentials["s3"]["public_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["s3"]["secret_key"]
os.environ["AWS_REGION"] = "eu-central-1"

# in total about 1 205 000 subdirectories -> 1k samples should be enough
def get_mean_std_for_norm(sample_size: int):
    urls = []
    dp_s3_urls = IterableWrapper(['s3://mi4people-soil-project/BigEarthNet-v1.0/']).list_files_by_s3().shuffle()
    for d in dp_s3_urls:
        name = os.path.dirname(d)
        if name not in urls:
            urls.append(name)
        if len(urls) == sample_size:
            break

    bands_pca_all = []
    for i in tqdm(urls):
        dp_s3_files = IterableWrapper([i+"/"]).list_files_by_s3().load_files_by_s3()
        bands = []
        for url, fd in dp_s3_files:
            file_name, extension = os.path.splitext(url)
            if extension == ".tif":
                tif_array = imageio.imread(fd)
                bands.append(tif_array)
        bands_resized = combine_and_resize_bands(bands, max_res=(120, 120))
        bands_pca = bands_resized.transpose(2, 0, 1).reshape(12, 14400)
        bands_pca_all.append(bands_pca)
    bands_pca_all = np.array(bands_pca_all)

    mean = []
    std = []
    for i in range(12):
        band_pca_i_all = bands_pca_all[:,i]
        percentile_1st = np.percentile(band_pca_i_all, 1)
        percentile_99th = np.percentile(band_pca_i_all, 99)
        mean.append(percentile_1st)
        std.append(percentile_99th-percentile_1st)
    return mean, std


def means_stds_analysis_plot(means, stds):
    # Calculate the average mean and standard deviation for the means
    avg_means = np.mean(means, axis=0)
    std_of_means = np.std(means, axis=0)

    # Calculate the average mean and standard deviation for the stds
    avg_stds = np.mean(stds, axis=0)
    std_of_stds = np.std(stds, axis=0)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting the error bars for means
    ax1.errorbar(range(1, len(avg_means) + 1), avg_means, yerr=std_of_means, fmt='o', capsize=5, color='tab:red')
    ax1.set_title('Error Analysis for Means of Bands')
    ax1.set_xlabel('Bands')
    ax1.set_ylabel('Mean Value')
    ax1.set_xticks(range(1, len(avg_means) + 1))

    # Plotting the error bars for standard deviations
    ax2.errorbar(range(1, len(avg_stds) + 1), avg_stds, yerr=std_of_stds, fmt='^', capsize=5, color='tab:blue')
    ax2.set_title('Error Analysis for Stds of Bands')
    ax2.set_xlabel('Bands')
    ax2.set_ylabel('Standard Deviation Value')
    ax2.set_xticks(range(1, len(avg_stds) + 1))

    plt.tight_layout()
    plt.savefig("means_stds_analysis.png")

    return avg_means, avg_stds

if __name__ == "__main__":
    num_runs = 5
    sample_size = 5 #1000
    means = []
    stds = []
    for i in range(num_runs):
        mean, std = get_mean_std_for_norm(sample_size)
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    final_means, final_stds = means_stds_analysis_plot(means, stds)
    np.save("means.npy", final_means)
    np.save("stds.npy", final_stds)
