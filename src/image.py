import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# This module is used to process image data for the MI estimation task. Note that
# much of the code here is non-functional without the associated data.

rho_values = { # These are the correlation values for the Gaussiam Markov fields
    "area": {"small": -0.12, "large": -0.227},
    "diffuse": {"small": -0.0012, "large": -0.00127712},
    "sparse": {"small": -0.045, "large": -0.11}
}

def get_pixel_numbers(image_height, image_width, region):

    # This function returns a grid that has been numbered
    # row-by-row from left to right.

    indices = np.reshape(np.arange(image_height*image_width), [image_height, image_width])
    (top, bottom, left, right) = region
    region_indices = indices[top:bottom, left:right]
    return region_indices.flatten()

def get_center_region(length, img_height, img_width):

    # This function extracts the four corners of a patch
    # located in the middle of the img_height x img_wdith
    # # image. 

    top = img_height // 2 - length // 2
    left = img_width // 2 - length // 2
    region = (top, top + length, left, left + length)
    return region

def get_area_law_cov(length, rho):

    # This function constructs a covariance matrix
    # with every variable being correlated with its
    # four nearest neighbors.

    q = np.eye(length**2)
    for i in range(length):
        for j in range(length):
            q_row = i * length + j
            for (m, l) in [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]:
                if (m < length and m >= 0) and (l < length and l >= 0):
                    q_col = m * length + l
                    q[q_row, q_col] = rho
    cov = np.linalg.inv(q)
    return cov

def get_diffuse_volume_cov(length, rho):

    # This function constructs a covariance matrix
    # with every variable equally correlated to all
    # other variables.

    q = np.full([length**2, length**2], rho)
    for i in range(length**2):
        q[i, i] = 1
    cov = np.linalg.inv(q)
    return cov

def get_sparse_volume_cov(length, rho):

    # This function constructs a covariance matrix
    # with sparse correlations that obey a volume law
    # in their scaling.

    gen = np.random.RandomState(123456789)
    q = np.eye(length**2)
    for i in range(length):
        for j in range(length):
            q_row = i * length + j
            for (m, l) in [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]:
                if (m < length and m >= 0) and (l < length and l >= 0):
                    q_col = m * length + l
                    q[q_row, q_col] = rho
    shuffle = gen.permutation(length**2)
    q = q[shuffle, :]
    q = q[:, shuffle]
    cov = np.linalg.inv(q)
    return cov

def get_marginal_entropy(cov, remove = [], keep = []):

    # This function computes the marginal entropy of a 
    # Gaussian Markov random field with respect to a specific
    # set of variables, selected by either keeping or removing
    # some of the variables.

    remove = np.asarray(remove)
    keep = np.asarray(keep)
    cov = np.asarray(cov)
    if remove.size != 0:
        red_cov = np.delete(np.delete(cov, remove, axis = 0), remove, axis = 1)
    elif keep.size != 0:
        red_cov = cov[keep]
        red_cov = red_cov[:, keep]
    else:
        red_cov = cov
    good_pixels = np.nonzero(np.greater(np.diagonal(red_cov), 10**-6))[0]
    valid_cov = red_cov[good_pixels]
    valid_cov = valid_cov[:, good_pixels]
    entropy = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * valid_cov)[1]
    return entropy

def get_gaussian_mutual_information(cov, variable_indices_a):

    # This function computes the MI for a Gaussian distribution
    # with the given covariance matrix.

    entropy_a = get_marginal_entropy(cov, keep = variable_indices_a)
    entropy_b = get_marginal_entropy(cov, remove = variable_indices_a)
    total_entropy = get_marginal_entropy(cov)
    mutual_information = entropy_a + entropy_b - total_entropy
    return mutual_information

def get_gaussian_fit(sample_images):

    # This function fits a Gaussian to a given sample of 
    # images.

    flat_images = np.reshape(sample_images, [sample_images.shape[0], -1])
    mean = np.mean(flat_images, axis = 0)
    cov = np.cov(flat_images, rowvar = False)
    return (cov, mean)

def get_analytic_MI(cov, image_shape, max_length):

    # This function computes the exact mutual information
    # between an inner and outer patch of variable in a 
    # Gaussian Markov random field, with patch sizes ranging
    # from 1 to max_length.

    mi = []
    for length in range(1, max_length):
        (height, width) = image_shape
        inner_region = get_center_region(length, height, width)
        inner_indices = get_pixel_numbers(height, width, inner_region)
        known_mi = get_gaussian_mutual_information(cov, inner_indices)
        mi.append(known_mi)
    return mi
    
def get_gaussian_images(cov, mean, num_images):

    # This function samples images from a Gaussian distribution
    # with the specified mean and covariance.

    length = int(cov.shape[0] ** 0.5)
    flat_images = np.random.multivariate_normal(mean = mean, cov = cov, size = [num_images], check_valid = 'raise')
    images = np.reshape(flat_images, [num_images, length, length])
    return images

def get_images(source, num_images, strength = "small"):

    # This function retrieves images from the specified dataset, as
    # well as the mean and covatiance from the Gaussian image sets.

    if source == 'tiny':
        images = np.load('tiny_images_100.npy')[:num_images]
        cov = cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1]*images.shape[2])
    elif source == 'mnist':
        ((train_images, _), (test_images, _)) = datasets.mnist.load_data()
        num_images = min(num_images, train_images.shape[0] + test_images.shape[0])
        images = np.concatenate([train_images, test_images], axis = 0)[:num_images] / 255
        cov = np.eye(images.shape[1] * images.shape[2])
        mean = np.zeros(images.shape[1]*images.shape[2])
    elif source == 'gauss_mnist':
        ((train_images, _), (test_images, _)) = datasets.mnist.load_data()
        mnist_images = np.concatenate([train_images, test_images], axis = 0) / 255
        (cov, mean) = get_gaussian_fit(mnist_images)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'gauss_tiny':
        cov = np.load("tiny_images_cov.npy")
        mean = np.zeros(cov.shape[0])
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'area':
        rho = rho_values["area"][strength]
        length = 28
        cov = get_area_law_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'diffuse':
        rho = rho_values["diffuse"][strength]
        length = 28
        cov = get_diffuse_volume_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    elif source == 'sparse':
        rho = rho_values["sparse"][strength]
        length = 28
        cov = get_sparse_volume_cov(length, rho)
        mean = np.zeros(length**2)
        images = get_gaussian_images(cov, mean, num_images)
    else:
        raise ValueError("Image source not recognized.")
    return (images, cov, mean)

def plot_cov():

    # This function plots the covariance matricies from
    # the strongly-correlated Gaussian Markov random fields
    # with respect to the center pixel, which is marked in 
    # red.

    fontsize = 16
    (_, ax_list) = plt.subplots(1, 3, figsize = (13, 4))
    for (i, scaling_type) in enumerate(["area", "diffuse", "sparse"]):
        (_, cov, _) = get_images(scaling_type, 1, strength = "large")
        cov[405, 405] = 0
        corr = cov[405].reshape([28, 28])
        ax_list[i].imshow(corr)
        for tick in ax_list[i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax_list[i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
    ax_list[0].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    ax_list[1].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    ax_list[2].add_patch(mpatches.Rectangle((12.55, 13.495), 0.89, 0.969, color = "red"))
    plt.tight_layout()
    plt.text(-0.15, 1.02, "a)", fontsize = fontsize + 6, transform = ax_list[0].transAxes)
    plt.text(-0.15, 1.02, "b)", fontsize = fontsize + 6, transform = ax_list[1].transAxes)
    plt.text(-0.15, 1.02, "c)", fontsize = fontsize + 6, transform = ax_list[2].transAxes)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.933, wspace=0.23, hspace=0)
    plt.savefig("scaling_covs.pdf")

def plot_tiny_mnist_cov_images():

    # This function samples images from the Gaussian distribution
    # fit to MNIST and the Tiny Images, and then plots them.

    (_, ax_list) = plt.subplots(2, 2, figsize = (6, 5))
    (mnist_images, mnist_cov, _) = get_images("gauss_mnist", 1)
    (tiny_images, tiny_cov, _) = get_images("gauss_tiny", 1)
    mnist_cov[405, 405] = 0
    tiny_cov[405, 405] = 0
    mnist_corr = mnist_cov[405].reshape([28, 28])
    tiny_corr = tiny_cov[405].reshape([28, 28])
    ax_list[0][1].imshow(mnist_corr)
    ax_list[0][0].imshow(tiny_corr)
    ax_list[1][1].imshow(mnist_images[0], cmap = "gray")
    ax_list[1][0].imshow(tiny_images[0], cmap = "gray")
    ax_list[0][1].set_title("MNIST")
    ax_list[0][0].set_title("Tiny Images")
    ax_list[0][0].add_patch(mpatches.Rectangle((12.6, 13.48), 0.77, 0.85, color = "red"))
    ax_list[0][1].add_patch(mpatches.Rectangle((12.6, 13.48), 0.77, 0.85, color = "red"))
    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0, hspace = None)
    plt.savefig("image_cov.pdf")

def plot_sampled_images():

    # This function samples an image from each of the six
    # Gaussian Markov random fields and then plots it.

    (_, ax_list) = plt.subplots(2, 3)
    for (i, size) in enumerate(["small", "large"]):
        for (j, scaling_type) in enumerate(["area", "diffuse", "sparse"]):
            (image, _, _) = get_images(scaling_type, 1, size)
            ax_list[i][j].imshow(image[0], cmap = "gray")
    ax_list[0][0].set_title("Nearest-Neighbor")
    ax_list[0][1].set_title("Uniform")
    ax_list[0][2].set_title("Randomized")
    ax_list[0][0].set_ylabel("Weak", fontsize = 12)
    ax_list[1][0].set_ylabel("Strong", fontsize = 12)
    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0)
    plt.savefig("gauss_samples.pdf")
        
def plot_gaussian():

    # This function computes analytic MI values for the 
    # Gaussian distributions that were fit to MNIST and 
    # the Tiny Images.

    fontsize = 14
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 1, figsize = (10, 6))

    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    size = 27
    lengths = list(range(1, size))
    (_, mnist_cov, _) = get_images("gauss_mnist", 1)
    tiny_cov = np.load("tiny_images_cov.npy")
    image_length = int(mnist_cov.size**(1/4))
    mnist_mi = get_analytic_MI(mnist_cov, [image_length, image_length], size + 1)
    tiny_mi = get_analytic_MI(tiny_cov, [image_length, image_length], size + 1)
    axes.plot(lengths, mnist_mi[:-1])
    axes.plot(lengths, tiny_mi[:-1])
    axes.set_xlabel('Partition Length (pixels)', fontsize = fontsize + 2)
    axes.set_ylabel('Mutual Information (nats)', fontsize = fontsize + 2)
    plt.legend(["Gaussian fit to Tiny Images", "Gaussian fit to MNIST"], fontsize = fontsize + 2)
    plt.tight_layout()
    plt.savefig("gaussian.pdf")

def plot_averages():

    # This function plots the MI estimates for the MNIST and
    # Tiny Images datasets.

    fontsize = 14
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 1, figsize = (10, 6))

    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    mnist_data_raw = np.load("averages/logistic_dense_mnist_0_1.0_70000.npy")
    mnist_trials = mnist_data_raw.reshape([20,27,2])
    tiny_data_raw = np.load("averages/logistic_dense_tiny_0_1.0_700000.npy")
    tiny_trials = tiny_data_raw.reshape([20,27,2])
    lengths = np.arange(1, mnist_trials.shape[1])
    mnist_mean = np.mean(mnist_trials, axis = 0)
    mnist_std = np.std(mnist_trials, axis = 0)
    tiny_mean = np.mean(tiny_trials, axis = 0)
    tiny_std = np.std(tiny_trials, axis = 0)
    axes.plot(lengths, mnist_mean[:26, 1])
    axes.plot(lengths, tiny_mean[:26, 1])
    axes.legend(["70,000 MNIST Images", "700,000 Tiny Images"], fontsize = fontsize + 2)
    axes.fill_between(lengths, (mnist_mean + mnist_std)[:26, 1], (mnist_mean - mnist_std)[:26, 1], alpha = 0.3)
    axes.fill_between(lengths, (tiny_mean + tiny_std)[:26, 1], (tiny_mean - tiny_std)[:26, 1], alpha = 0.3)
    axes.set_xlabel("Partition Length (pixels)", fontsize = fontsize + 2)
    axes.set_ylabel("Mutual Information (nats)", fontsize = fontsize + 2)
    axes.legend(["70,000 MNIST Images", "700,000 Tiny Images"], fontsize = fontsize + 2)
    plt.tight_layout()
    plt.savefig("mnist_tiny.pdf")

def plot_large_small_avg(scaling_type):

    # This function creates plots of the MI for the 
    # Gaussian Markov random fields, using the analytic
    # value and the three estimates with different sample
    # sizes.

    if scaling_type not in ["area", "diffuse", "sparse"]:
        raise ValueError("Scaling type '{}' not recognized.".format(scaling_type))
    size = 27

    # Compute the average MI values and the standard deviation
    # for the small correlaton value.

    small_corr = rho_values[scaling_type]["small"]
    small_0 = np.load('trials/logistic_dense_{}_{}_1.0_70000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:20]
    small_1 = np.load('trials/logistic_dense_{}_{}_1.0_700000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:10]
    small_2 =  np.load('trials/logistic_dense_{}_{}_1.0_7000000.npy'.format(scaling_type, small_corr)).reshape([-1,27,2])[:5]
    small_mean_0 = np.mean(small_0, axis = 0)
    small_mean_1 = np.mean(small_1, axis = 0)
    small_mean_2 = np.mean(small_2, axis = 0)
    small_std_0 = np.std(small_0, axis = 0)
    small_std_1 = np.std(small_1, axis = 0)
    small_std_2 = np.std(small_2, axis = 0)
    (_, small_cov, _) = get_images(scaling_type, 1, strength = "small")
    small_length = int(small_cov.size**(1/4))
    small_exact = get_analytic_MI(small_cov, [small_length, small_length], size + 1)
    
    # Compute the average MI values and the standard deviation
    # for the large correlaton value.

    large_corr = rho_values[scaling_type]["large"]
    large_0 = np.load('trials/logistic_dense_{}_{}_1.0_70000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:20]
    large_1 = np.load('trials/logistic_dense_{}_{}_1.0_700000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:10]
    large_2 =  np.load('trials/logistic_dense_{}_{}_1.0_7000000.npy'.format(scaling_type, large_corr)).reshape([-1,27,2])[:5]
    large_mean_0 = np.mean(large_0, axis = 0)
    large_mean_1 = np.mean(large_1, axis = 0)
    large_mean_2 = np.mean(large_2, axis = 0)
    large_std_0 = np.std(large_0, axis = 0)
    large_std_1 = np.std(large_1, axis = 0)
    large_std_2 = np.std(large_2, axis = 0)
    (_, large_cov, _) = get_images(scaling_type, 1, strength = "large")
    large_length = int(large_cov.size**(1/4))
    large_exact = get_analytic_MI(large_cov, [large_length, large_length], size + 1)

    # Plot the mean values and standard deviation on a pair
    # of axes, and then save the result.

    lengths = list(range(1, size))
    fontsize = 14
    linewidth = 2
    plt.rc("axes", linewidth = 1)
    (_, axes) = plt.subplots(1, 2, figsize = (12, 6))

    for tick in axes[0].xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)
    for tick in axes[0].yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)

    for tick in axes[1].xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)
    for tick in axes[1].yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize + 1)

    handles = [mlines.Line2D([], [], color = "C0"), mlines.Line2D([], [], color = "C1"), 
        mlines.Line2D([], [], color = "C2"), mlines.Line2D([], [], color = "C3")]
    axes[0].legend(handles, [r'$7\times10^4$ samples', r'$7\times10^5$ samples', r"$7\times10^6$ samples", 'Exact'],
    fontsize = fontsize + 2).set_zorder(-1)

    axes[0].plot(lengths, small_mean_0[:26, 0], "C0", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_0 + small_std_0)[:26, 0], (small_mean_0 - small_std_0)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_mean_1[:26, 0], "C1", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_1 + small_std_1)[:26, 0], (small_mean_1 - small_std_1)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_mean_2[:26, 0], "C2", linewidth = linewidth)
    axes[0].fill_between(lengths, (small_mean_2 + small_std_2)[:26, 0], (small_mean_2 - small_std_2)[:26, 0], alpha = 0.3)
    axes[0].plot(lengths, small_exact[:26], "C3", linewidth = linewidth)
    axes[0].set_xlabel('Partition Length (L)', fontsize = fontsize + 3)
    axes[0].set_ylabel('Mutual Information (nats)', fontsize = fontsize + 3)

    axes[1].plot(lengths, large_mean_0[:26, 0], "C0", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_0 + large_std_0)[:26, 0], (large_mean_0 - large_std_0)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_mean_1[:26, 0], "C1", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_1 + large_std_1)[:26, 0], (large_mean_1 - large_std_1)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_mean_2[:26, 0], "C2", linewidth = linewidth)
    axes[1].fill_between(lengths, (large_mean_2 + large_std_2)[:26, 0], (large_mean_2 - large_std_2)[:26, 0], alpha = 0.3)
    axes[1].plot(lengths, large_exact[:26], "C3", linewidth = linewidth)
    axes[1].set_xlabel('Partition Length (L)', fontsize = fontsize + 3)

    plt.tight_layout()
    plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.2, hspace = None)
    plt.savefig(f"{scaling_type}.pdf")
