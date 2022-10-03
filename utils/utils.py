import cv2


class VideoFrameIndexError(IndexError):
    def __init__(self, message="frame number out of bounds. The video has less frames."):
        super().__init__(message)


def frame_generator(filename):
    """Generator that yields frames from a video.

    Parameters
    ----------
    filename : string
        name of the video file.

    Yields
    -------
    array
        the current video frame. For color video, the channel order is
        RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist.
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {filename} not found!')

    ret, frame = cap.read()
    while(ret):
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cap.read()


def get_frame(filename, frame_number=0):
    """Get a specific frame from a video.

    Parameters
    ----------
    filename : string
        name of the video file.
    frame_number : int, optional
        which frame to return, by default 0.

    Returns
    -------
    array
        the frame corresponding to ``frame_number``. For color video,
        the channel order is RGB.

        Returns ``None`` if ``frame_number`` is larger than the total
        number of frames in the video or if the video file is not
        found.

    Raises
    ------
    VideoFrameIndexError
        if ``frame_number`` is greater than the number of frames in the video.
    """
    for i, frame in enumerate(frame_generator(filename)):
        if i == frame_number:
            return frame

    raise VideoFrameIndexError
