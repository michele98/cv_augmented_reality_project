import cv2

from utils.matchers import FeatureMatcher


class VideoFrameIndexError(IndexError):
    """Raised when the wanted video frame is out of bounds."""
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


def overlay_ar(frame, reference_frame, ar_layer, ar_mask):
    """Overlay the AR layer onto the video frame. The overlay is done
    originally on the reference frame, then a homography between the
    reference frame and the original frame is computed using local
    invariant features matching.
    
    The local invariant features are found using SIFT, the matches are
    found using FLANN KDTree, and finally the homography is computed
    on the resulting matches using RANSAC.

    Parameters
    ----------
    frame : image
        original video frame.
    reference_frame : image
        reference frame onto which the AR layer is initially projected.
        Its resolution must be the same as ``original_frame``.
    ar_layer : image
        image that needs to be overlaid onto the video.
        Its resolution must be the same as ``original_frame``.
    ar_mask : image
        mask for the AR layer.
        Its resolution must be the same as ``original_frame``.

    Returns
    -------
    image
        the image with the overlaid AR layer.
    """
    h, w = frame.shape[0], frame.shape[1]

    matcher = FeatureMatcher(reference_frame, frame)
    matcher.find_matches()
    H, _ = matcher.get_homography()

    ar_layer_warped = cv2.warpPerspective(ar_layer, H, dsize=(w, h))
    ar_mask_warped = cv2.warpPerspective(ar_mask, H, dsize=(w, h))

    ar_frame = frame.copy()
    ar_frame[ar_mask_warped==255] = ar_layer_warped[ar_mask_warped==255]

    return ar_frame
