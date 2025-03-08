"""
Module that contains standalone functions for imaging analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce
import plotly.graph_objects as go

def plot_axis(image, principal_axis, centroid):
    """
    Plots the image with the principal axis overlayed.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        principal_axis (numpy.ndarray): Array containing the principal axis vector [x, y].
        centroid (tuple): Coordinates of the centroid (mean_x, mean_y).
    """
    height, width = image.shape
    mean_x, mean_y = centroid

    # Principal axis vector components
    dx, dy = principal_axis

    # Normalize the principal axis to scale across the image dimensions
    if abs(dx) > abs(dy):
        scale = width / (2 * abs(dx))
    else:
        scale = height / (2 * abs(dy))

    x_start = mean_x - dx * scale
    x_end = mean_x + dx * scale
    y_start = mean_y - dy * scale
    y_end = mean_y + dy * scale

    # Clip the line to image boundaries
    x_start, x_end = np.clip([x_start, x_end], 0, width)
    y_start, y_end = np.clip([y_start, y_end], 0, height)

    # Plot the image
    plt.imshow(image, cmap="viridis", origin="lower", extent=(0, width, 0, height))

    # Overlay the principal axis
    plt.plot(
        [x_start, x_end],
        [y_start, y_end],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Principal Axis",
    )

    # Mark the centroid
    plt.scatter(mean_x, mean_y, color="blue", label="Centroid")

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Image with Principal Axis")
    plt.legend()

    plt.show()


def plot_intensity_contour(image, grid_x, grid_y, grid_z):
    """
    Plots the original image with the grid interpolation overlayed.

    Parameters:
        image (numpy.ndarray): 2D array representing the original image.
        grid_x (numpy.ndarray): X-coordinates of the interpolated grid.
        grid_y (numpy.ndarray): Y-coordinates of the interpolated grid.
        grid_z (numpy.ndarray): Interpolated intensity values from the grid.
    """
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(
        image,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Intensity")
    plt.title("Original Image")

    # Plot the interpolated spline
    plt.subplot(1, 2, 2)
    plt.imshow(
        grid_z,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Interpolated Intensity")
    plt.contour(grid_x, grid_y, grid_z, levels=10, colors="red", linewidths=0.5)
    plt.title("Grid Interpolation")

    # Add common labels
    plt.suptitle("Image with Grid Interpolation", fontsize=16)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.tight_layout()

    plt.show()


def plot_spline(image, x_spline, y_spline):
    """
    Plots the original image with the principal axis spline overlayed.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        x_spline (numpy.ndarray): Interpolated x-coordinates of the spline.
        y_spline (numpy.ndarray): Interpolated y-coordinates of the spline.
    """
    plt.figure(figsize=(8, 8))

    # Plot the original image
    plt.imshow(
        image,
        cmap="viridis",
        origin="lower",
        extent=(0, image.shape[1], 0, image.shape[0]),
    )
    plt.colorbar(label="Intensity")

    # Overlay the spline principal axis
    plt.plot(
        x_spline,
        y_spline,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Principal Axis Spline",
    )

    # Add labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Image with Principal Axis Spline")
    plt.legend()

    plt.show()


def plot_bounding_box(image: np.ndarray, bounding_box: tuple):
    """
    Plot the image with the bounding box overlayed.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    bounding_box (tuple): The bounding box coordinates (min_y, min_x, max_y, max_x).
    """
    if bounding_box is None:
        print("No bounding box to plot.")
        return

    min_y, min_x, max_y, max_x = bounding_box

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="viridis")
    plt.gca().add_patch(
        plt.Rectangle(
            (min_x, min_y),
            max_x - min_x + 1,
            max_y - min_y + 1,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
    )
    plt.title("Image with Bounding Box")
    plt.show()


def plot_binary_image(image: np.ndarray):
    """
    Plot an image where all non-zero intensities are set to 1.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    """
    binary_image = (image > 0).astype(int)
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_image, cmap="gray")
    plt.title("Non-zero Intensities (Binary Representation)")
    plt.show()


def plot_3d(image: np.ndarray) -> None:
    """
    Plots a 3D graph where the z-height represents the intensity of the image.

    Parameters:
        image (numpy.ndarray): 2D array representing the image, where each element is the intensity.

    Returns:
        None
    """
    # Get the dimensions of the image
    height, width = image.shape

    # Create a grid of x, y coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with intensity as z-height
    ax.plot_surface(x, y, image, cmap="viridis", edgecolor="none")

    # Label the axes
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Intensity (Z-height)")
    ax.set_title("3D Intensity Plot")

    # Show the plot
    plt.show()

def plot_voxels(R, downsample_factor=4):

    # Find nonzero indices
    nonzero_indices = np.nonzero(R)

    # Bounding box limits
    xmin, xmax = nonzero_indices[0].min(), nonzero_indices[0].max()
    ymin, ymax = nonzero_indices[1].min(), nonzero_indices[1].max()
    zmin, zmax = nonzero_indices[2].min(), nonzero_indices[2].max()

    # Crop R to bounding box
    R_cropped = R[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    factor = downsample_factor
    R_downsampled = block_reduce(R_cropped, block_size=(factor, factor, factor), func=np.max)

    voxel_size = 0.019  # Each voxel is 0.019mm

    fig = go.Figure(data=go.Volume(
        x=(np.linspace(xmin, xmax, R_downsampled.shape[0]) * voxel_size).repeat(R_downsampled.shape[1] * R_downsampled.shape[2]),
        y=(np.tile(np.linspace(ymin, ymax, R_downsampled.shape[1]).repeat(R_downsampled.shape[2]), R_downsampled.shape[0]) * voxel_size),
        z=(np.tile(np.linspace(zmin, zmax, R_downsampled.shape[2]), R_downsampled.shape[0] * R_downsampled.shape[1]) * voxel_size),
        value=R_downsampled.flatten(),
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis',
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X [mm]',
        yaxis_title='Y [mm]',
        zaxis_title='Z [mm]',
        aspectmode='data'
    ), title='Downsampled 3D Volume Plot')

    fig.show()


def plot_voxels_axis(R, downsample_factor=4, principal_axis=None, centroid=None):
    """
    Plots downsampled voxel intensities and optionally a principal axis and centroid.

    Parameters:
        R (np.ndarray): 3D intensity matrix.
        downsample_factor (int): Factor by which the voxel matrix is downsampled.
        principal_axis (np.ndarray, optional): Principal axis vector.
        centroid (np.ndarray, optional): Centroid coordinates of the voxel distribution.
    """

    # Find nonzero indices
    nonzero_indices = np.nonzero(R)

    # Bounding box limits
    xmin, xmax = nonzero_indices[0].min(), nonzero_indices[0].max()
    ymin, ymax = nonzero_indices[1].min(), nonzero_indices[1].max()
    zmin, zmax = nonzero_indices[2].min(), nonzero_indices[2].max()

    # Crop R to bounding box
    R_cropped = R[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    factor = downsample_factor
    R_downsampled = block_reduce(R_cropped, block_size=(factor, factor, factor), func=np.max)


    fig = go.Figure(data=go.Volume(
        x=np.linspace(xmin, xmax, R_downsampled.shape[0]).repeat(R_downsampled.shape[1] * R_downsampled.shape[2]),
        y=np.tile(np.linspace(ymin, ymax, R_downsampled.shape[1]).repeat(R_downsampled.shape[2]), R_downsampled.shape[0]),
        z=np.tile(np.linspace(zmin, zmax, R_downsampled.shape[2]), R_downsampled.shape[0] * R_downsampled.shape[1]),
        value=R_downsampled.flatten(),
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis',
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ), title='Downsampled 3D Volume Plot')


    # Plot principal axis if provided
    if principal_axis is not None and centroid is not None:
        scale = np.linalg.norm(R.shape) / 2
        line_points = np.array([
            centroid - principal_axis * scale,
            centroid + principal_axis * scale
        ])
        fig.add_trace(go.Scatter3d(
            x=line_points[:, 0], y=line_points[:, 1], z=line_points[:, 2],
            mode='lines',
            line=dict(color='red', width=8),
            name='Principal Axis'
        ))

        # Plot centroid
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Centroid'
        ))

    fig.show()




def plot_3d_projections(cam_image, ito_image):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot test_cam on the xy plane
    x_cam, y_cam = np.meshgrid(np.arange(cam_image.shape[1]), np.arange(cam_image.shape[0]))
    ax.plot_surface(x_cam, y_cam, np.zeros_like(cam_image), rstride=1, cstride=1, facecolors=plt.cm.viridis(cam_image / np.max(cam_image)), shade=False)

    # Plot test_ito on the xz plane
    x_ito, z_ito = np.meshgrid(np.arange(ito_image.shape[1]), np.arange(ito_image.shape[0]))
    ax.plot_surface(x_ito, np.zeros_like(ito_image), z_ito, rstride=1, cstride=1, facecolors=plt.cm.viridis(ito_image / np.max(ito_image)), shade=False)

    # Plot a black yz surface with white text "No Readout"
    y_no_readout, z_no_readout = np.meshgrid(np.arange(ito_image.shape[0]), np.arange(ito_image.shape[0]))
    ax.plot_surface(np.zeros_like(y_no_readout), y_no_readout, z_no_readout, color='#440154', shade=False)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Rotate the plot for better visibility
    ax.view_init(elev=30, azim=45)

    plt.show()