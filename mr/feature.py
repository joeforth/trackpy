# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import re
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from scipy.ndimage import morphology
from scipy.ndimage import filters
from scipy.ndimage import fourier
from scipy.ndimage import measurements
from scipy.ndimage import interpolation
from scipy import stats
from utils import memo
import sql
import diagnostics
import _Cfilters

logger = logging.getLogger(__name__)

def subtract_junk(image, junk_image):
    """When there is junk in the field of view, call this
    function first to subtract a baseline image from each frame."""
    image -= junk_image # in place

def bandpass(image, lshort, llong):
    """Convolve with a Gaussian to remove short-wavelength noise,
    and subtract out long-wavelength variations,
    retaining features of intermediate scale."""
    if not 2*lshort < llong:
        raise ValueError, ("The smoothing length scale must be more" 
                           "than twice the noise length scale.")
    smoothed_background = filters.uniform_filter(image, 2*llong+1)
    result = np.fft.ifft2(fourier.fourier_gaussian(np.fft.fft2(image), lshort))
    result -= smoothed_background
    # Where result < 0 that pixel is definitely not a feature. Zero to simplify.
    return result.real.clip(min=0.)

@memo
def circular_mask(diameter, side_length=None):
    """A circle of 1's inscribed in a square of 0's,
    the 'footprint' of the features we seek."""
    r = int(diameter)/2
    L = side_length if side_length else int(diameter)
    points = np.arange(-int(L/2), int(L/2) + 1)
    x, y = np.meshgrid(points, points)
    z = np.sqrt(x**2 + y**2)
    mask = np.zeros_like(z, dtype='bool')
    mask[z < r] = 1
    return mask

@memo
def _rgmask(diameter):
    r = int(diameter)/2
    points = np.arange(-r, r + 1)
    x, y = np.meshgrid(points, points)
    mask = x**2 + y**2
    mask[mask > r**2] = 0
    mask += (1/6.) # Right?
    return mask

@memo
def _thetamask(diameter):
    r = int(diameter)/2
    return circular_mask(diameter) * \
        np.fromfunction(lambda y, x: np.arctan2(r-y,x-r), (diameter, diameter)) 

@memo
def _sinmask(diameter):
    return circular_mask(diameter)*np.sin(2*_thetamask(diameter))

@memo
def _cosmask(diameter):
    return circular_mask(diameter)*np.cos(2*_thetamask(diameter))

def _local_maxima(image, diameter, separation, percentile=64):
    """Find local maxima whose brightness is above a given percentile."""
    # Find the threshold brightness, representing the given
    # percentile among all NON-ZERO pixels in the image.
    flat = np.ravel(image)
    threshold = stats.scoreatpercentile(flat[flat > 0], percentile)
    # The intersection of the image with its dilation gives local maxima.
    assert image.dtype == np.uint8, "Perform dilation on exact (uint8) data." 
    dilation = morphology.grey_dilation(
        image, footprint=circular_mask(diameter, separation))
    maxima = np.where((image == dilation) & (image > threshold))
    if not np.size(maxima) > 0:
        raise ValueError, ("Bad image! Found zero maxima above the {}"
                           "-percentile treshold at {}.".format(
                           percentile, threshold))
    # Flat peaks, for example, return multiple maxima.
    # Eliminate redundancies within the separation distance.
    maxima_map = np.zeros_like(image)
    maxima_map[maxima] = image[maxima]
    peak_map = filters.generic_filter(
        maxima_map, _Cfilters.nullify_secondary_maxima(), 
        footprint=circular_mask(separation), mode='constant')
    # Also, do not accept peaks near the edges.
    margin = int(separation)/2
    peak_map[..., :margin] = 0
    peak_map[..., -margin:] = 0
    peak_map[:margin, ...] = 0
    peak_map[-margin:, ...] = 0
    peaks = np.where(peak_map != 0)
    if not np.size(peaks) > 0:
        raise ValueError, "Bad image! All maxima were in the margins."
    # Return coords in as a numpy array, shaped so it can be passed directly
    # to the DataFrame constructor.
    return np.array([peaks[1], peaks[0]]).T # columns: x, y

def _estimate_mass(image, x, y, diameter):
    "Compute the total brightness in the neighborhood of a local maximum."
    r = int(diameter)/2
    x0 = x - r
    x1 = x + r + 1
    y0 = y - r
    y1 = y + r + 1
    neighborhood = circular_mask(diameter)*image[y0:y1, x0:x1]
    return np.sum(neighborhood)

def _refine_centroid(image, x, y, diameter, minmass=100, iterations=10):
    """Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness. Return its coordinates, integrated
    brightness, size (Rg), and eccentricity (0=circular)."""
    # Define the square neighborhood of (x, y).
    r = int(diameter)/2
    x0, y0 = x - r, y - r
    x1, y1 = x + r + 1, y + r + 1
    neighborhood = circular_mask(diameter)*image[y0:y1, x0:x1]
    yc, xc = measurements.center_of_mass(neighborhood)  # neighborhood coords
    yc, xc = yc + y0, xc + x0  # image coords
    ybounds = (0, image.shape[0] - 1 - 2*r)
    xbounds = (0, image.shape[1] - 1 - 2*r)
    if iterations < 1:
        raise ValueError, "Set iterations=1 or more."
    for iteration in xrange(iterations):
        if (xc + r - x0 < 0.1 and yc + r - y0 < 0.1):
            break  # Accurate enough.
        # Start with whole-pixel shifts.
        if abs(xc - x0 - r) >= 0.6:
            x0 = np.clip(round(xc) - r, *xbounds)
            x1 = x0 + 2*r + 1
        if abs(yc - y0 -r) >= 0.6:
            y0 = np.clip(round(yc) - r, *ybounds)
            y1 = y0 + 2*r + 1
#       if abs(xc - x0 - r) < 0.6 and (yc -y0 -r) < 0.6:
            # Subpixel interpolation using a second-order spline.
#           interpolation.shift(neighborhood,[yc, xc],mode='constant',cval=0., order=2)
        neighborhood = circular_mask(diameter)*image[y0:y1, x0:x1]    
        yc, xc = measurements.center_of_mass(neighborhood)  # neighborhood coordinates
        yc, xc = yc + y0, xc + x0  # image coords
    
    # Characterize the neighborhood of our final centroid.
    mass = np.sum(neighborhood)    
    Rg = np.sqrt(np.sum(_rgmask(diameter)*image[y0:y1, x0:x1])/mass)
    ecc = np.sqrt((np.sum(neighborhood*_cosmask(diameter)))**2 + 
                  (np.sum(neighborhood*_sinmask(diameter)))**2) / \
                  (mass - neighborhood[r, r] + 1e-6)
    return Series([xc, yc, mass, Rg, ecc])

def _locate_centroids(image, diameter, separation=None, 
                      percentile=64, minmass=100, pickN=None):
    """Find bright Guassian-like blobs against a dark background.
    See wrapper function locate() for descriptions of the parameters
    and the returned values."""
    # Check parameters.
    if not diameter & 1:
        raise ValueError, "Feature diameter must be an odd number. Round up."
    if not separation:
        separation = diameter + 1
    image = (255./image.max()*image.clip(min=0.)).astype(np.uint8)
    c = DataFrame(
        _local_maxima(image, diameter, separation, percentile=percentile),
        columns=['x', 'y'])
    approx_mass = c.apply(
        lambda x: _estimate_mass(image, x[0], x[1], diameter), axis=1)
    refined_c = c[approx_mass > minmass].apply(
        lambda x: _refine_centroid(image, x[0], x[1], diameter, minmass), 
        axis=1)
    refined_c.columns = ['x', 'y', 'mass', 'size', 'ecc']
    logger.info("%s local maxima, %s of qualifying mass", 
                len(c), len(refined_c)) 
    return refined_c 

def locate(image_file, diameter, minmass=100., separation=None, 
           noise_size=1, smoothing_size=None, invert=True, junk_image=None,
           percentile=64, pickN=None):
    """Read an image, do optional image preparation and cleanup, and locate 
    Gaussian-like blobs of a given size above a given total brightness.

    Parameters
    ----------
    image_file : string file path
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a 
       crucial parameter for elminating spurrious features.
    separation : feature separation in px
    noise_size : scale of Gaussian blurring. Default 1.
    smoothing_size : defauls to separation
    invert : Set to True if features are darker than background. Default True.
    junk_image : an image that will be subtracted from each frame before
        it is processed
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    pickN : Not Implemented

    Returns
    -------
    DataFrame([x, y, mass, size, ecc])

    where mass means total integrated brightness of the blob
    and size means the radius of gyration of its Gaussian-like profile
    """
    smoothing_size = smoothing_size if smoothing_size else diameter # default
    image = plt.imread(image_file)
    if isinstance(junk_image,str):
        if type(junk_image) is str:
            junk_image = plt.imread(junk_image)
        subtract_junk(image, junk_image)  
    if invert:
        # Efficient way of doing image = 1 - image
        image *= -1; image += 1 
    image = bandpass(image, noise_size, smoothing_size)
    f = _locate_centroids(image, diameter, separation=separation,
                          percentile=percentile, minmass=minmass,
                          pickN=pickN)
    return f

def batch(trial, stack, images, diameter, minmass=100, separation=None,
          noise_size=1, smoothing_size=None, invert=True, junk_image=None,
          percentile=64, pickN=None, override=False):
    """Process a list of images, doing optional image preparation and cleanup, 
    locating Gaussian-like blobs of a given size above a given total brightness.

    Parameters
    ----------
    trial : integer Trial Number or string Trial Name
    stack : integer Stack Number or string Stack Name
    image_file : string file path
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a 
       crucial parameter for elminating spurrious features.
    separation : feature separation in px
    noise_size : scale of Gaussian blurring. Default 1.
    smoothing_size : defauls to separation
    invert : Set to True if features are darker than background. Default True.
    junk_image : an image that will be subtracted from each frame before
        it is processed
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    pickN : Not Implemented

    Returns
    -------
    DataFrame([x, y, mass, size, ecc])

    where mass means total integrated brightness of the blob
    and size means the radius of gyration of its Gaussian-like profile
    """
    images = _cast_images(images)
    conn = sql.connect()
    if sql.feature_duplicate_check(trial, stack, conn):
        if override:
            logger.info('Overriding')
        else:
            logging.error('There are entries for this trial and stack already.')
            conn.close()
            return False
    junk_image = plt.imread(junk_image)
    for frame, filepath in enumerate(images):
        frame += 1 # Start at 1, not 0.
        centroids = locate(filepath, diameter, minmass, separation, 
                           noise_size, smoothing_size, invert, junk_image,
                           percentile, pickN)
        sql.insert_feat(trial, stack, frame, centroids, conn, override)
        logger.info("Completed Trial %s Stack %s Frame %s", 
                    trial, stack, frame)
    conn.close()

def sample(images, diameter, minmass=100, separation=None,
           noise_size=1, smoothing_size=None, invert=True, junk_image=None,
           percentile=64, pickN=None):
    """Try parameters on the first, middle, and last image in a stack.
    See notes for explanation of plots.

    Parameters
    ----------
    images : directory, list of files, or one file
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a 
       crucial parameter for elminating spurrious features.
    separation : feature separation in px
    noise_size : scale of Gaussian blurring. Default 1.
    smoothing_size : defauls to separation
    invert : Set to True if features are darker than background. Default True.
    junk_image : an image that will be subtracted from each frame before
        it is processed
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    pickN : Not Implemented

    Returns
    -------
    None

    Notes
    -----
    For each image, the features are circled.
    For the last image, the decimal part of the x and y positions are
    histogrammed. When you are achieving subpixel accuracy, these histograms
    should be flat. If they dip in the middle, try increasing the diameter
    parameter.
    """
    images = _cast_images(images)
    get_elem = lambda x, indicies: [x[i] for i in indicies]
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
    axes = axes[0][0], axes[0][1], axes[1][0], axes[1][1] # flat
    fig.set_size_inches(12, 8)
    if len(images) < 3:
        samples = images
    else:
        # first, middle, last
        samples = get_elem(images, [0, len(images)/2, -1])
    for i, image_file in enumerate(samples):
        logger.info("Sample %s of %s...", 1+i, len(samples))
        f = locate(image_file, diameter, minmass, separation,
                   noise_size, smoothing_size, invert, junk_image,
                   percentile, pickN)
        diagnostics.annotate(image_file, f, ax=axes[i])
        if i == len(samples) - 1:
            fig, ax = plt.subplots()
            f[['x', 'y']].applymap(lambda x: x % 1).hist(ax=ax)
            fig, ax = plt.subplots()
            ax.plot(f['mass'], f['ecc'], 'ko', alpha=0.3)
            ax.set_xlabel('mass')
            ax.set_ylabel('ecc')

def _cast_images(images):
    """Accept a list of image files, a directory of image files, 
    or a single image file. Return contents as a list of strings."""
    if type(images) is list:
        return images
    elif type(images) is str:
        if os.path.isfile(images):
            return list(images) # a single-element list
        elif os.path.isdir(images):
            images = list_images(images)
            return images
    else:
        raise TypeError, ("images must be a directory path, a file path, or "
                          "a list of file paths.")

def list_images(directory):
    "List the path to all image files in a directory."
    files = os.listdir(directory)
    images = [os.path.join(directory, f) for f in files if \
        os.path.isfile(os.path.join(directory, f)) and re.match('.*\.png', f)]
    if not images: logging.error('No images!')
    return sorted(images)

