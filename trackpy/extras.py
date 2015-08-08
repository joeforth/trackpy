"""These functions generate handy plots."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import zip
from itertools import tee
from collections import Iterable
from functools import wraps
import warnings

import numpy as np

from .utils import print_update


__all__ = ['an_save', 'an3d_save']


def make_axes(func):
    """
    A decorator for plotting functions.
    NORMALLY: Direct the plotting function to the current axes, gca().
              When it's done, make the legend and show that plot.
              (Instant gratificaiton!)
    BUT:      If the uses passes axes to plotting function, write on those axes
              and return them. The user has the option to draw a more complex
              plot in multiple steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            # Delete legend keyword so remaining ones can be passed to plot().
            try:
                legend = kwargs['legend']
            except KeyError:
                legend = None
            else:
                del kwargs['legend']
            result = func(*args, **kwargs)
            if not (kwargs['ax'].get_legend_handles_labels() == ([], []) or \
                    legend is False):
                plt.legend(loc='best')
            plt.show()
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def make_fig(func):
    """See make_axes."""
    import matplotlib.pyplot as plt
    wraps(func)
    def wrapper(*args, **kwargs):
        if 'fig' not in kwargs:
            kwargs['fig'] = plt.gcf()
            func(*args, **kwargs)
            plt.show()
        else:
            return func(*args, **kwargs)
    return wrapper


def _normalize_kwargs(kwargs, kind='patch'):
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == 'line2d':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          mec='markeredgecolor', mew='markeredgewidth',
                          mfc='markerfacecolor', ms='markersize',)
    elif kind == 'patch':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          ec='edgecolor', fc='facecolor',)
    for short_name in long_names:
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs


@make_axes
def an_save(centroids, image, circle_size=None, color=None,
             invert=False, ax=None, split_category=None, split_thresh=None,
             imshow_style={}, plot_style={}, ppm=1, filename='Filename here'):
    """Modification of the annotate function - saves each image and scales the axes.

    Parameters
    ----------
    centroids : DataFrame including columns x and y
    image : image array (or string path to image file)
    circle_size : Deprecated.
        This will be removed in a future version of trackpy.
        Use `plot_style={'markersize': ...}` instead.
    color : single matplotlib color or a list of multiple colors
        default None
    invert : If you give a filepath as the image, specify whether to invert
        black and white. Default True.
    ax : matplotlib axes object, defaults to current axes
    split_category : string, parameter to use to split the data into sections
        default None
    split_thresh : single value or list of ints or floats to split
        particles into sections for plotting in multiple colors.
        List items should be ordered by increasing value.
        default None
    imshow_style : dictionary of keyword arguments passed through to
        the `Axes.imshow(...)` command the displays the image
    plot_style : dictionary of keyword arguments passed through to
        the `Axes.plot(...)` command that marks the features

    Returns
    ------
    axes
    """
    import matplotlib.pyplot as plt
    print('Running modified code')
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[-1] in (3, 4)):
        raise ValueError("image has incorrect dimensions. Please input a 2D "
                         "grayscale or RGB(A) image. For 3D image annotation, "
                         "use annotate3d. Multichannel images can be "
                         "converted to RGB using pims.display.to_rgb.")

    if circle_size is not None:
        warnings.warn("circle_size will be removed in future version of "
                      "trackpy. Use plot_style={'markersize': ...} instead.")
        if 'marker_size' not in plot_style:
            plot_style['marker_size'] = np.sqrt(circle_size)  # area vs. dia.
        else:
            raise ValueError("passed in both 'marker_size' and 'circle_size'")

    _plot_style = dict(markersize=15, markeredgewidth=2,
                       markerfacecolor='none', markeredgecolor='r',
                       marker='o', linestyle='none')
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)
    _imshow_style.update(imshow_style)

    # https://docs.python.org/2/library/itertools.html
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    if color is None:
        color = ['r']
    if isinstance(color, six.string_types):
        color = [color]
    if not isinstance(split_thresh, Iterable):
        split_thresh = [split_thresh]

    # The parameter image can be an image object or a filename.
    if isinstance(image, six.string_types):
        image = plt.imread(image)
    if invert:
        ax.imshow(1-image, **_imshow_style)
    else:
        ax.imshow(image, **_imshow_style)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)
    #ax.invert_yaxis()
    # Rescale our axis labels
    y_ax = np.shape(image)[0]
    x_ax = np.shape(image)[1]
    y_tic_locs = []
    y_tic_labs = []
    x_tic_locs = []
    x_tic_labs = []
    for i in range(0, y_ax+1, int(y_ax/5)):
        y_tic_locs.append(i)
        y_tic_labs.append(np.floor(i/ppm))
    for i in range(0, x_ax+1, int(x_ax/5)):
        x_tic_locs.append(i)
        x_tic_labs.append(np.floor(i/ppm))
    plt.xticks(x_tic_locs, x_tic_labs)
    plt.yticks(y_tic_locs, y_tic_labs)
    ax.set_xlabel('x (\xb5m)')
    ax.set_ylabel('y (\xb5m)')

    if split_category is None:
        if np.size(color) > 1:
            raise ValueError("multiple colors specified, no split category "
                             "specified")
        _plot_style.update(markeredgecolor=color[0])
        ax.plot(centroids['x'], centroids['y'],
                **_plot_style)
    else:
        if len(color) != len(split_thresh) + 1:
            raise ValueError("number of colors must be number of thresholds "
                             "plus 1")
        low = centroids[split_category] < split_thresh[0]
        _plot_style.update(markeredgecolor=color[0])
        ax.plot(centroids['x'][low], centroids['y'][low],
                **_plot_style)

        for c, (bot, top) in zip(color[1:-1], pairwise(split_thresh)):
            indx = ((centroids[split_category] >= bot) &
                    (centroids[split_category] < top))
            _plot_style.update(markeredgecolor=c)
            ax.plot(centroids['x'][indx], centroids['y'][indx],
                    **_plot_style)

        high = centroids[split_category] >= split_thresh[-1]
        _plot_style.update(markeredgecolor=color[-1])
        ax.plot(centroids['x'][high], centroids['y'][high],
                **_plot_style)
    # Save the file here
    plt.savefig(filename)
    return ax


def an3d_save(centroids, image, file='stupid', scale = 1, **kwargs):
    """
    An extension of annotate that annotates a 3D image and returns a scrollable
    stack for display in IPython. Parameters: see annotate.
    """
    import matplotlib.pyplot as plt
    from pims.display import scrollable_stack

    if image.ndim != 3 and not (image.ndim == 4 and image.shape[-1] in (3, 4)):
        raise ValueError("image has incorrect dimensions. Please input a 3D "
                         "grayscale or RGB(A) image. For 2D image annotation, "
                         "use annotate. Multichannel images can be "
                         "converted to RGB using pims.display.to_rgb.")

    if kwargs.get('ax') is None:
        kwargs['ax'] = plt.gca()

    for i, imageZ in enumerate(image):
        centroidsZ = centroids[np.logical_and(centroids['z'] > i - 0.5,
                                              centroids['z'] < i + 0.5)]
        an_save(centroidsZ, imageZ, filename = file + '_' + str(i), ppm = scale, **kwargs)
        plt.cla()