import cv2
import numpy as np

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
    cap.release()


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


def overlay_ar(frame, reference_frame, ar_layer, ar_mask, return_homography=False):
    """Overlay the AR layer onto the video frame. The overlay is done}
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
    return_homography : bool
        if True, the homography of the warping between the frame and
        the reference frame is returned. By default False.

    Returns
    -------
    image
        the image with the overlaid AR layer.
    image, 3x3 array
        the image with the overlaid AR layer
        the homography between the original frame and the reference
        frame. Only if ``return_homography`` is set to True.
    """
    h, w = frame.shape[0], frame.shape[1]

    matcher = FeatureMatcher(reference_frame, frame)
    matcher.find_matches()
    H, _ = matcher.get_homography()

    ar_layer_warped = cv2.warpPerspective(ar_layer, H, dsize=(w, h))
    ar_mask_warped = cv2.warpPerspective(ar_mask, H, dsize=(w, h))

    ar_frame = frame.copy()
    ar_frame[ar_mask_warped==255] = ar_layer_warped[ar_mask_warped==255]

    if return_homography:
        return ar_frame, H
    return ar_frame


def save_ar_video(filename_src, filename_dst, ar_layer, ar_mask, reference_frame=None):
    """Save the AR overlaid video with the F2R (frame to reference)
    method.

    Parameters
    ----------
    filename : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
        Its resolution must be the same as ``original_frame``.
    ar_mask : image
        mask for the AR layer.
        Its resolution must be the same as ``original_frame``.
    reference_frame : image
        reference frame onto which the AR layer is initially projected.
        Its resolution must be the same as ``original_frame``.
    """
    if reference_frame is None:
        reference_frame = next(frame_generator(filename_src))

    out = cv2.VideoWriter(filename_dst, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    for i, frame in enumerate(frame_generator(filename_src)):
        print(f"writing frame {i}", end = '\r')
        ar_frame = overlay_ar(frame, reference_frame, ar_layer, ar_mask)
        out.write(cv2.cvtColor(ar_frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"writing frame {i}")
    print('done.')


def save_ar_video_f2f(filename_src, filename_dst, ar_layer, ar_mask, reference_frame=None, drift_correction_step=0):
    """Save the AR overlaid video with the F2F (frame to frame) method.

    Parameters
    ----------
    filename : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
        Its resolution must be the same as ``original_frame``.
    ar_mask : image
        mask for the AR layer.
        Its resolution must be the same as ``original_frame``.
    reference_frame : image
        reference frame onto which the AR layer is initially projected.
        Its resolution must be the same as ``original_frame``.
    drift_correction_step : int, optional
        this much passes until the homography is recomputed using the
        original reference frame. This is done to prevent drifting of
        the AR layer. Setting this to 1 makes this effectively a F2R
        matching method.
    """
    if reference_frame is None:
        reference_frame = next(frame_generator(filename_src))
    if drift_correction_step < 0:
        raise ValueError("the drift correction step must be positive.")

    out = cv2.VideoWriter(filename_dst, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    H_original = np.eye(3)
    ar_layer_original = ar_layer
    ar_mask_original = ar_mask
    reference_frame_original = reference_frame

    H = H_original
    for i, frame in enumerate(frame_generator(filename_src)):
        print(f"writing frame {i}", end = '\r')

        # reset the referernce frame to avoid excessive drift of the AR layer
        if drift_correction_step>0 and i%drift_correction_step==0:
            H = H_original
            ar_layer = ar_layer_original
            ar_mask = ar_mask_original
            reference_frame = reference_frame_original

        ar_frame, H_ar = overlay_ar(frame, reference_frame, ar_layer, ar_mask, return_homography=True)
        reference_frame = frame

        H = H@H_ar
        h, w = ar_frame.shape[0], ar_frame.shape[1]
        ar_layer = cv2.warpPerspective(ar_layer_original, H, dsize = (w, h))
        ar_mask = cv2.warpPerspective(ar_mask_original, H, dsize = (w, h))
        out.write(cv2.cvtColor(ar_frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"writing frame {i}")
    print('done.')
