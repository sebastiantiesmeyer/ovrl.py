import numpy as np
from scipy.ndimage import maximum_filter
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numba 
import matplotlib.pyplot as plt
# create circular kernel:

def create_circular_kernel(r):
    """
    Creates a circular kernel of radius r.
    Parameters
    ----------
    r : int
        The radius of the kernel.

    Returns
    -------
    kernel : np.array
        A 2d array of the circular kernel.

    """
    
    span = np.linspace(-1,1,r*2)
    X,Y = np.meshgrid(span,span)
    return (X**2+Y**2)**0.5<=1

def get_kl_divergence(p,q):
    # mask = (p!=0) * (q!=0)
    output = np.zeros(p.shape)
    # output[mask] = p[mask]*np.log(p[mask]/q[mask])
    output[:] = p[:]*np.log(p[:]/q[:])
    return output

def determine_localmax(distribution, min_distance=3, min_expression=5):
    """
    Returns a list of local maxima in a kde of the data frame.
    Parameters
    ----------
    distribution : np.array
        A 2d array of the distribution.
    min_distance : int, optional
        The minimum distance between local maxima. The default is 3.
    min_expression : int, optional
        The minimum expression level to include in the histogram. The default is 5.

    Returns
    -------
    rois_x : list
        A list of x coordinates of local maxima.
    rois_y : list
        A list of y coordinates of local maxima.

    """
    localmax_kernel = create_circular_kernel(min_distance)
    localmax_projection = (distribution == maximum_filter(
        distribution, footprint=localmax_kernel))

    rois_x, rois_y = np.where((distribution > min_expression) & localmax_projection)

    return rois_x, rois_y, distribution[rois_x, rois_y]

## These functions are going to be seperated into a package of their own at some point:

from sklearn.decomposition import PCA as Dimred

def haversine_to_rgb(coords):
    #project to unit sphere:
    x = np.sin(coords[:, 0]) * np.cos(coords[:, 1])
    y = np.sin(coords[:, 0]) * np.sin(coords[:, 1])
    z = np.cos(coords[:, 0])

    coords = np.array([x,y,z]).T

    print(coords.shape)

    #project to unit cube:
    coords = coords/np.max(np.abs(coords),axis=1,keepdims=True)

    return coords/2+0.5

def rotate_points(data):
    # Convert the angles to radians

    
    # Create the rotation matrix that rotates rgb values to cover the color space efficiently:
    rotation_matrix = np.array([[-1.,  0.,  1.],
                                [-1.,  1., -1.],
                                [-1., -0.,  1.]])
    
    # Perform the rotation
    rotated_data = data#np.dot(data, rotation_matrix)
    
    return rotated_data

def fill_color_axes(rgb,dimred=None):

    if dimred is None:
        dimred = Dimred(n_components=3)
        dimred.fit(rgb)

    facs = dimred.transform(rgb)

    # rotate the facs to cover color axes efficiently:
    facs = rotate_points(facs)

    # return facs,dimred
    return facs,dimred


# create circular kernel:
def create_circular_kernel(kernel_width):
    span = np.linspace(-1,1,kernel_width)
    X,Y = np.meshgrid(span,span)
    return (X**2+Y**2)**0.5<=1


# normalize array:
def min_to_max(arr):
    arr=arr-arr.min(0,keepdims=True)
    arr/=arr.max(0,keepdims=True)
    return arr

# define a function that fits expression data to into the umap embeddings:
def transform_embeddings(expression,pca,embedder_2d,embedder_3d):

    factors = pca.transform(expression)
    # embedding_color = embedder_3d.transform(factors)
    embedding_color = embedder_3d.transform(factors)
    
    embedding = embedder_2d.transform(factors)
    # embedding_color = (embedding_color-color_min)/(color_max-color_min)
    
    return embedding, embedding_color

# define a function that plots the embeddings, with celltype centers rendered as plt.texts on top:
def plot_embeddings(embedding,embedding_color,celltype_centers,celltypes,rasterized=False):
    colors = np.clip(embedding_color.copy(),0,1)

    plt.scatter(embedding[:,0],embedding[:,1],c=(colors),alpha=0.1,marker='.',rasterized=rasterized)

    text_artists = []
    for i in range(len(celltypes)):
        cog_x,cog_y = (celltype_centers[i,0],celltype_centers[i,1])
        if not any(np.isnan([cog_x,cog_y])):
            t = plt.text(cog_x,cog_y,celltypes[i],color='k',fontsize=6)
            text_artists.append(t)

    untangle_text(text_artists)

def untangle_text(text_artists,max_iterations=10000):

    ax = plt.gca()
    # ax_size = ax.get_window_extent().size
    inv = ax.transData.inverted()

    artist_coords = np.array([text_artist.get_position() for text_artist in text_artists])
    artist_coords = artist_coords+np.random.normal(0,0.001,artist_coords.shape)
    artist_extents = ([text_artist.get_window_extent() for text_artist in text_artists])
    artist_edges = np.array([inv.transform(extent.get_points()) for extent in artist_extents])
    artist_extents = (artist_edges[:,1]-artist_edges[:,0])*0.5
    artist_coords = artist_coords+artist_extents

    # print(artist_extents[50],artist_coords[50],artist_edges.shape)
    # plt.scatter(*artist_coords.T,c='r',marker='x')
    # plt.scatter(*(artist_coords+artist_extents*1).T  ,c='g',marker='x')

    # initial_artist_coords = artist_coords.copy()

    for i in range(100):

        
        relative_positions_x = (artist_coords[:,0][:,None]-artist_coords[:,0][None,:])
        relative_positions_y = (artist_coords[:,1][:,None]-artist_coords[:,1][None,:])

        relative_distances_x = np.abs(relative_positions_x)
        relative_distances_y = np.abs(relative_positions_y)

        relative_distances_x = relative_distances_x-artist_extents[:,0]
        relative_distances_y = relative_distances_y-artist_extents[:,1]
        relative_distances_x = relative_distances_x-artist_extents[:,None,0] 
        relative_distances_y = relative_distances_y-artist_extents[:,None,1]
        

        intersection = (relative_distances_x<0)&(relative_distances_y<0)
        
        
        velocities_x = np.zeros_like(relative_positions_x)
        velocities_y = np.zeros_like(relative_positions_y)

        velocities_x += relative_positions_x*intersection
        velocities_y += relative_positions_y*intersection
       
       
        velocities_x[np.eye(velocities_x.shape[0],dtype=bool)]=0
        velocities_y[np.eye(velocities_y.shape[0],dtype=bool)]=0 

        delta = np.stack([velocities_x,velocities_y],axis=1).sum(-1)

        artist_coords = artist_coords + delta*0.07
       
       
    for i,text_artist in enumerate(text_artists):
        text_artist.set_position(artist_coords[i,:]-artist_extents[i,:])

# define a function that subsamples spots around x,y given a window size:
def get_spatial_subsample_mask(coordinate_df,x,y,plot_window_size=5):
    return (coordinate_df.x>x-plot_window_size)&(coordinate_df.x<x+plot_window_size)&(coordinate_df.y>y-plot_window_size)&(coordinate_df.y<y+plot_window_size)

# define a function that returns the k nearest neighbors of x,y:
def create_knn_graph(coords,k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices

# get a kernel-weighted average of the expression values of the k nearest neighbors of x,y:
def get_knn_expression(distances,neighbor_indices,genes, gene_labels,bandwidth=2.5):

    weights = np.exp(-distances/bandwidth)
    local_expression = pd.DataFrame(index = genes, columns = np.arange(distances.shape[0])).astype(float)

    for i,gene in enumerate(genes):
        weights_ = weights.copy()
        weights_[(gene_labels[neighbor_indices])!=i] = 0
        local_expression.loc[gene,:] = weights_.sum(1)
    
    return local_expression

@numba.njit(fastmath=True)
def discounted_euclidean_grad(x, y,discount=0.5):
    r"""Standard euclidean distance and its gradient.
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """

    x_centered = x - y
    result = np.sum(x_centered ** 2)
    d = np.power(result,discount)
    grad = 2*discount* x_centered * result**(1e-6 * (discount-1))

    return d, grad