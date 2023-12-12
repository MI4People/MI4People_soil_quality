from torchdata.datapipes.iter import IterableWrapper
from src.data.img_engineering import get_first_n_pcs
from src.data.ingestion import combine_and_resize_bands
import os
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["AWS_ACCESS_KEY_ID"] = "AKIASUMSMN7USSB6MHHC"
os.environ["AWS_SECRET_ACCESS_KEY"] = "Wj1iF/9RrE9qKg2beKuxN0sWRDRhmuxiIbcEuNqB"
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
        bands_pca = get_first_n_pcs(bands_resized, 3)
        bands_pca = bands_pca.reshape(3, -1)
        bands_pca_all.append(bands_pca)
    bands_pca_all = np.array(bands_pca_all)

    mean = []
    std = []
    for i in range(3):
        band_pca_i_all = bands_pca_all[:,i]
        percentile_1st = np.percentile(band_pca_i_all, 1)
        percentile_99th = np.percentile(band_pca_i_all, 99)
        mean.append(percentile_1st)
        std.append(percentile_99th-percentile_1st)
    return mean, std

if __name__ == "__main__":
    num_runs = 10
    sample_size = 1000
    means = []
    stds = []
    for i in range(num_runs):
        mean, std = get_mean_std_for_norm(sample_size)
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    np.save("means.npy", means)
    np.save("stds.npy", stds)

    plot_m_means = np.mean(means, axis=0)
    plot_e_means = np.std(means, axis=0)
    plt.errorbar(["Band 1", "Band 2", "Band 3"], plot_m_means, yerr=plot_e_means, fmt='o', capsize=5)
    plt.title('Error Analysis for means of PCA bands')
    plt.xlabel('PCA bands')
    plt.ylabel('Mean Value')
    plt.savefig("means_analysis.png")

    plt.cla()

    plot_m_stds = np.mean(stds, axis=0)
    plot_e_stds = np.std(stds, axis=0)
    plt.errorbar(["Band 1", "Band 2", "Band 3"], plot_m_stds, yerr=plot_e_stds, fmt='o', capsize=5)
    plt.title('Error Analysis for stds of PCA bands')
    plt.xlabel('PCA bands')
    plt.ylabel('Standard deviation Value')
    plt.savefig("stds_analyses.png")
