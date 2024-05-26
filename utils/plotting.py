import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy.typing as npt
from IPython.display import HTML


def display_video_grid(
    videos: npt.NDArray,
    grid_size: tuple | list,
    figsize: tuple | list,
    norm: str = "linear",
) -> HTML:
    """
    Display a grid of grayscale videos in a Jupyter Notebook.

    Parameters
    ----------
    videos : numpy.ndarray
        A 4D numpy array of shape (num_videos, num_frames, height,
        width), representing the grayscale videos to be displayed.
    grid_size : Tuple[int, int]
        The number of rows and columns in the grid.
    figsize : Tuple[int, int]
        The size of the figure in inches.
    norm : str, optional
        The normalization to apply to the images, by default "linear".

    Returns
    -------
    IPython.display.HTML
        An HTML object that renders the animation in the Jupyter Notebook.
    """

    fig, axes = plt.subplots(*grid_size, figsize=figsize)
    axes = axes.flatten()

    ims = []
    for i, video in enumerate(videos):
        ims.append(axes[i].imshow(video[0], cmap="gray", norm=norm))
        axes[i].axis("off")

    def update(frame):
        for im, video in zip(ims, videos):
            im.set_array(video[frame % len(video)])
        return ims

    ani = animation.FuncAnimation(
        fig, update, frames=range(len(videos[0])), blit=True, repeat=True
    )
    plt.close(fig)  # Prevents the static image from showing up in the notebook
    return HTML(ani.to_jshtml())
