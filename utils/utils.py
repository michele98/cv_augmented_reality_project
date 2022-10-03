import cv2


def frame_generator(filename):
    """Generator that yields frames from a video.

    Parameters
    ----------
    filename : string
        name of the video file

    Yields
    -------
    array
        the current video frame. For color video, the channel order is
        RGB.
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f'Video file {filename} not found!')
        return

    ret, frame = cap.read()
    while(ret):
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cap.read()


def get_frame(filename, frame_number=0):
    """Get a specific frame from a video.

    Parameters
    ----------
    filename : string
        name of the video file
    frame_number : int, optional
        which frame to return, by default 0

    Returns
    -------
    array
        the frame corresponding to ``frame_number``. For color video,
        the channel order is RGB.

        Returns ``None`` if ``frame_number`` is larger than the total
        number of frames in the video or if the video file is not
        found.
    """
    for i, frame in enumerate(frame_generator(filename)):
        if i == frame_number:
            return frame

    print('There are less frames than frame_number!')
    return None
