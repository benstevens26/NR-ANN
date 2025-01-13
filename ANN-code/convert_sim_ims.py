import glob
import math

import numpy as np
from IPython.display import HTML, display
from scipy.signal import convolve2d

display(HTML("<style>.container { width:100% !important; }</style>"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def calc_light_fraction(dist, QE, f=25, N=0.85, reflect=False, ref_GEM="Cu"):
    L = 0.5 * (1 - dist / math.sqrt((0.5 * f / N) ** 2 + dist * dist))
    if reflect:
        if ref_GEM == "Cu":
            factor = 0.69
        elif ref_GEM == "Ni":
            factor = 0.65
        L += factor * 0.67 * 0.33 * L * (118.7 / (118.7 + 2 + 2 * 0.57)) ** 2
    return L * QE * 0.34


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


light_fraction = calc_light_fraction(118.7, 0.23, f=25, N=0.85, reflect=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def quick_convert_im(im, dark_sample, light_fraction, reflect_fraction):
    im = light_fraction * 10 * im / 0.11
    # im = im + reflect_fraction*convolve2d(im,gauss_kernel(min(im.shape),60),mode='same',boundary='symm')
    im = np.round(im) + dark_sample  # * 0.11

    return im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def convert_im(
    im, dark_sample, light_fraction=light_fraction, reflect_fraction=0, gain_corr=10
):
    gen = np.random.default_rng()

    # noise? causing negative entries into image. going to zero any negative entries
    # im[im < 0] = 0

    #     im = light_fraction*gain_corr*im/0.11
    im = gen.binomial((gain_corr * im).astype(np.int32), light_fraction) / 0.11
    if reflect_fraction:
        im = im + reflect_fraction * convolve2d(
            im, gauss_kernel(10, 5), mode="same", boundary="symm"  # type: ignore
        )
    im = np.round(im) + dark_sample  # * 0.11

    return im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def load_dark_stack(num=0):
    return np.load(f"darks/quest_std_dark_{num}.npy")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def bin_im(im, binning):  # can handle stack of images
    if len(im.shape) == 3:  # stack of images
        return np.sum(
            [
                im[:, i::binning, j::binning]
                for i in range(binning)
                for j in range(binning)
            ],
            axis=0,
        )
    else:  # individual image
        return np.sum(
            [
                im[i::binning, j::binning]
                for i in range(binning)
                for j in range(binning)
            ],
            axis=0,
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def create_md_and_noise(binning=1, save=True):
    darks = load_dark_stack(np.random.randint(0, 10))
    darks = bin_im(darks, binning)
    md = np.mean(darks, axis=0)
    noise = np.std(darks, axis=0)

    darks[np.abs(darks - md) > 5 * noise] = np.nan

    md = np.nanmean(darks, axis=0)
    noise = np.nanstd(darks, axis=0)

    darks[np.abs(darks - md) > 5 * noise] = np.nan

    md = np.nanmean(darks, axis=0)
    noise = np.nanstd(darks, axis=0)

    if save:
        np.save(f"darks/master_dark_{binning}x{binning}.npy", md)
        np.save(f"darks/noise_{binning}x{binning}.npy", noise)

    return md, noise


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Write some code to sample the dark images and convert the simulated images using the function above.
# Once you have the sample of each dark image loaded you will need to sample the SAME REGION of the master dark image and subtract it.
# The (dark_sample[X,Y] - master_dark[X,Y]) is what should go into the 'dark_sample' argument in the convert_im() function above.

# Each master dark and noise file details the binning of the image. adding 2x2 pixels in a square grid will reduce the dimensionality of the image and improve the signal-to-noise.
# If you choose to bin the pixels and improve signal to noise, you will need to use the appropriate master dark.

# You might choose to use the noises in some way when extracting parameters or in the CNNs; have a think.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def get_dark_sample(m_dark, im_dims, example_dark):
    """
    Gets a random image-sized sample of the example_dark and subtracts a corresponding sample of the master dark to leave just the noise.
    :param m_dark: 2D array containing the master dark.
    :param im_dims: Array containing the dimensions of the event image as [x,y].
    :param example_dark: 2D array containing the example dark event.
    :returns dark_sample: 2D image-sized array of the randomly sampled noise.
    """

    m_dark_dims = [len(m_dark[0]), len(m_dark)]

    sample_start = [
        np.random.randint(0, m_dark_dims[0] - im_dims[0]),
        np.random.randint(0, m_dark_dims[1] - im_dims[1]),
    ]

    m_dark_sample = [
        m_dark[sample_start[1] + i][sample_start[0] : sample_start[0] + im_dims[0]]
        for i in range(0, im_dims[1])
    ]

    example_dark_sample = [
        example_dark[sample_start[1] + i][
            sample_start[0] : sample_start[0] + im_dims[0]
        ]
        for i in range(0, im_dims[1])
    ]

    dark_sample = np.array(example_dark_sample) - np.array(m_dark_sample)
    return dark_sample
