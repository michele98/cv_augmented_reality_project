import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.utils import frame_generator, overlay_ar, VideoFrameIndexError


def display(frame, i, refresh_output=True):
    if refresh_output: clear_output(wait = True)
    plt.imshow(frame)
    plt.gca().set_axis_off()
    plt.tight_layout(pad=0)
    plt.show()
    print(f"frame {i}")


def display_frames(filename, starting_frame=0, refresh_output=True):
    """Display video frames from a video file, starting from a
    starting frame.

    Parameters
    ----------
    filename : string
        name of the video file.
    starting_frame : int, optional
        by default 0.
    refresh_output : bool, optional
        choose whether each new frame clears the previous one,
        by default True.

    Raises
    ------
    VideoFrameIndexError
        if ``starting_frame`` is greater than the number of frames in the video.
    """

    for i, frame in enumerate(frame_generator(filename)):
        if i < starting_frame:
            continue
        try:
            display(frame, i, refresh_output)
        except KeyboardInterrupt:
            display(frame, i, refresh_output)
            break

    if i < starting_frame:
        raise VideoFrameIndexError
    print('Capture released')


def display_ar_frames(filename, ar_layer, ar_mask, reference_frame=None, starting_frame=0, refresh_output=True):
    """Show the AR overlaid video frame by frame.

    Parameters
    ----------
    filename : string
        name of the video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
        Its resolution must be the same as the video.
    ar_mask : image
        mask for the AR layer.
    reference_frame : image, optional
        if not provided, the first frame of the video is chosen as
        reference.
    starting_frame : int, optional
        frame of the video from which to start, by default 0.
    refresh_output : bool, optional
        choose whether each new frame clears the previous one,
        by default True.

    Raises
    ------
    VideoFrameIndexError
        if ``starting_frame`` is greater than the number of frames in the video.
    """
    if reference_frame is None:
        reference_frame = next(frame_generator(filename))

    for i, frame in enumerate(frame_generator(filename)):
        if i < starting_frame:
            continue
        ar_frame = overlay_ar(frame, reference_frame, ar_layer, ar_mask)

        try:
            display(ar_frame, i, refresh_output)
        except KeyboardInterrupt:
            display(ar_frame, i, refresh_output)
            break

    if i < starting_frame:
        raise VideoFrameIndexError
    print('Capture released')
