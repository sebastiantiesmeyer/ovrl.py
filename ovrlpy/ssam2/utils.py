### utils file with basic functions needed for the SSAM algorithm.

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.feature import peak_local_max


def kde_2d(coordinates, name=None, size=None, bandwidth=1.5):
    """
    Create a histogram of the data.
    """

    if coordinates.shape[0] == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size)

    if size is None:
        size = np.ceil(np.max(coordinates, axis=0)).astype(int)
        # print('size:',size)

    output = np.zeros(size)

    x_, _x = int(np.min(coordinates[:, 0])), int(np.ceil(np.max(coordinates[:, 0])))
    y_, _y = int(np.min(coordinates[:, 1])), int(np.ceil(np.max(coordinates[:, 1])))

    x_bins = np.linspace(x_, _x, _x - x_ + 1)
    y_bins = np.linspace(y_, _y, _y - y_ + 1)

    histogram, bins = np.histogramdd(
        [coordinates[:, 0], coordinates[:, 1]], bins=[x_bins, y_bins]
    )

    kde = gaussian_filter(histogram, sigma=bandwidth)

    output[
        int(bins[0].min()) : int(bins[0].max()), int(bins[1].min()) : int(bins[1].max())
    ] = kde
    # print(output.sum())

    return output


def kde_3d(coordinates, size=None, bandwidth=1.5):
    """
    Create a histogram of the data.
    """

    if coordinates.shape[0] == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size)

    if size is None:
        size = np.ceil(np.max(coordinates, axis=0)).astype(int)
        # print('size:',size)

    output = np.zeros(size)

    x_, _x = int(np.min(coordinates[:, 0])), int(np.ceil(np.max(coordinates[:, 0])))
    y_, _y = int(np.min(coordinates[:, 1])), int(np.ceil(np.max(coordinates[:, 1])))
    z_, _z = int(np.min(coordinates[:, 2])), int(np.ceil(np.max(coordinates[:, 2])))

    x_bins = np.linspace(x_, _x, _x - x_ + 1)
    y_bins = np.linspace(y_, _y, _y - y_ + 1)
    z_bins = np.linspace(z_, _z, _z - z_ + 1)

    histogram, bins = np.histogramdd(
        [coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]],
        bins=[x_bins, y_bins, z_bins],
    )

    kde = gaussian_filter(histogram, sigma=bandwidth)

    output[
        int(bins[0].min()) : int(bins[0].max()),
        int(bins[1].min()) : int(bins[1].max()),
        int(bins[2].min()) : int(bins[2].max()),
    ] = kde
    # print(output.sum())

    return output


def find_local_maxima(vf, min_pixel_distance: int = 5, min_expression: float = 2):
    """
    Find local maxima in a vector field.
    """
    local_maxima = peak_local_max(
        vf,
        min_distance=min_pixel_distance,
        threshold_abs=min_expression,
        exclude_border=True,
    )

    return local_maxima


def kde_and_sample(coordinates, sampling_coordinates, size=None, bandwidth=1.5):
    """
    Create a kde of the data and sample at 'sampling_coordinates'.
    """

    # coordinates+=bandwidth
    # sampling_coordinates+=bandwidth

    sampling_coordinates = np.round(sampling_coordinates).astype(int)

    if coordinates.shape[-1] == 2:
        kde = kde_2d(coordinates, size=size, bandwidth=bandwidth)
        output = kde[sampling_coordinates[:, 0], sampling_coordinates[:, 1]]
    elif coordinates.shape[-1] == 3:
        kde = kde_3d(coordinates, size=size, bandwidth=bandwidth)
        output = kde[
            sampling_coordinates[:, 0],
            sampling_coordinates[:, 1],
            sampling_coordinates[:, 2],
        ]

    return output


def crosscorr(x, y):
    """
    Calculate the cross-correlation between two matrices.
    """
    x -= np.array(x.mean(1))[:, None]
    y -= np.array(y.mean(1))[:, None]
    c = (np.dot(x, y.T) / x.shape[1]).squeeze()

    return np.nan_to_num(
        np.nan_to_num(c / np.array(x.std(1))[:, None]) / np.array(y.std(1))[None, :]
    )


def fill_celltypemaps(
    ct_map, fill_blobs=True, min_blob_area=0, filter_params={}, output_mask=None
):
    """
    Post-filter cell type maps created by `map_celltypes`.

    :param min_r: minimum threshold of the correlation.
    :type min_r: float
    :param min_norm: minimum threshold of the vector norm.
        If a string is given instead, then the threshold is automatically determined using
        sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
    :type min_norm: str or float
    :param fill_blobs: If true, then the algorithm automatically fill holes in each blob.
    :type fill_blobs: bool
    :param min_blob_area: The blobs with its area less than this value will be removed.
    :type min_blob_area: int
    :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
        Not used when `min_norm` is float.
    :type filter_params: dict
    :param output_mask: If given, the cell type maps will be filtered using the output mask.
    :type output_mask: np.ndarray(bool)
    """

    filtered_ctmaps = np.zeros_like(ct_map) - 1

    for cidx in np.unique(ct_map):
        mask = ct_map == cidx
        if min_blob_area > 0 or fill_blobs:
            blob_labels = measure.label(mask, background=0)
            for bp in measure.regionprops(blob_labels):
                if min_blob_area > 0 and bp.filled_area < min_blob_area:
                    for c in bp.coords:
                        mask[c[0], c[1]] = 0

                    continue
                if fill_blobs and bp.area != bp.filled_area:
                    minx, miny, maxx, maxy = bp.bbox
                    mask[minx:maxx, miny:maxy] |= bp.filled_image

        filtered_ctmaps[
            np.logical_and(mask == 1, np.logical_or(ct_map == -1, ct_map == cidx))
        ] = cidx

    return filtered_ctmaps
