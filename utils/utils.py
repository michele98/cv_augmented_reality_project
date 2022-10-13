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


def overlay_ar(frame, homography, ar_layer, ar_mask=None):
    """Overlay the AR layer onto the video frame using the provided
    homography.

    Parameters
    ----------
    frame : image
        original video frame.
    homography : 3x3 array
        homography for the warping of the AR layer onto the frame.
    ar_layer : image
        image that needs to be overlaid onto the video.
    ar_mask : image, optional
        mask for the AR layer.
        Its resolution must be the same as ``ar_layer``.

    Returns
    -------
    image
        the image with the overlaid AR layer.
    """
    if ar_mask is None:
        ar_mask = np.ones(ar_layer.shape, dtype=np.uint8)*255

    h, w = frame.shape[0], frame.shape[1]

    ar_layer_warped = cv2.warpPerspective(ar_layer, homography, dsize=(w, h))
    ar_mask_warped = cv2.warpPerspective(ar_mask, homography, dsize=(w, h))

    ar_frame = frame.copy()
    ar_frame[ar_mask_warped==255] = ar_layer_warped[ar_mask_warped==255]

    return ar_frame


def save_ar_video(filename_src, filename_dst, ar_layer, ar_mask=None, reference_image=None, drift_correction_step=0, start_frame=0, stop_frame=0, fps=30):
    """Save the AR overlaid video with the F2F (frame to frame) method,
    where the reference frame can be reset after a certain number of
    frames. If the correction is done at every frame, this effectively
    becomes F2R.

    Parameters
    ----------
    filename : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
    ar_mask : image, optional
        mask for the AR layer.
        Its resolution must be the same as ``ar_layer``.
    reference_image : image, optional
        reference image onto which the AR layer is projected.
        If not provided, the first video frame is used.
        It must have the same size as ``ar_layer``.
    drift_correction_step : int, optional
        this much passes until the homography is recomputed using the
        original reference frame. This is done to prevent drifting of
        the AR layer. Setting this to 1 makes this effectively a F2R
        matching method.
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than ``start_frame``.
    fps : int, optional
        frames per second of the video, by default 30.
    """
    frame_gen = frame_generator(filename_src) #initialize frame generator

    #go to the first wanted frame
    for i, frame in enumerate(frame_gen):
        first_frame = frame
        if i==start_frame:
            break

    if reference_image is None:
        reference_image = first_frame
    if drift_correction_step < 0:
        raise ValueError("the drift correction step must be positive.")
    if start_frame >= stop_frame and stop_frame > 0:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    # create the VideoWriter object
    # and set the resolution of the output video as the one of the input video
    h, w = first_frame.shape[0], first_frame.shape[1]
    out = cv2.VideoWriter(filename=filename_dst,
                          fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                          fps=fps,
                          frameSize=(w, h))

    # instantiate a FeatureMatcher for the reference image and the first frame
    matcher = FeatureMatcher(reference_image, first_frame)
    matcher.find_matches()

    # get keypoints and SIFT descriptors for the first video frame
    # this is done to save computation time later
    reference_keypoints, first_keypoints = matcher.get_keypoints()
    reference_descriptors, first_descriptors = matcher.get_descriptors()

    # compute the initial homography between the reference image and the first video frame
    first_H, _ = matcher.get_homography()
    H_history = first_H

    # setup the process for f2f
    previous_frame = first_frame
    previous_keypoints = first_keypoints
    previous_descriptors = first_descriptors

    for i, frame in enumerate(frame_gen, start=start_frame):
        if stop_frame > 0 and i==stop_frame:
            break

        print(f"writing frame {i}", end = '\r')
        # reset homography history after a fixed number of frames
        if drift_correction_step>0 and i%drift_correction_step==0:
            matcher = FeatureMatcher(reference_image, frame)
            matcher.set_descriptors_1(reference_keypoints, reference_descriptors)
            H_history = np.eye(3)       # reset homography history
        else:
            matcher = FeatureMatcher(previous_frame, frame)
            matcher.set_descriptors_1(previous_keypoints, previous_descriptors)

        # find the homography between the previous frame and the current one
        matcher.find_matches()
        H, _ = matcher.get_homography()
        H_history = H_history@H # update the homography history

        # overlay the frame with the ar layer
        ar_frame = overlay_ar(frame, H_history, ar_layer, ar_mask)

        # reset previous keypoints and frame for reuse in the next loop
        previous_frame = frame
        _, previous_keypoints = matcher.get_keypoints()
        _, previous_descriptors = matcher.get_descriptors()

        out.write(cv2.cvtColor(ar_frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"writing frame {i}")
    print('done.')


def save_ar_video_f2r(*args, **kwargs):
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
    ar_mask : image, optional
        mask for the AR layer.
        Its resolution must be the same as ``ar_layer``.
    reference_image : image, optional
        reference image onto which the AR layer is projected.
        If not provided, the first video frame is used.
        It must have the same size as ``ar_layer``.
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than ``start_frame``.
    fps : int, optional
        frames per second of the video, by default 30.
    """
    return save_ar_video(drift_correction_step=1, *args, **kwargs)


def save_ar_video_f2f(*args, **kwargs):
    """Save the AR overlaid video with the F2F (frame to frame) method.

    Parameters
    ----------
    filename : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
    ar_mask : image, optional
        mask for the AR layer.
        Its resolution must be the same as ``ar_layer``.
    reference_image : image, optional
        reference image onto which the AR layer is projected.
        If not provided, the first video frame is used.
        It must have the same size as ``ar_layer``.
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than ``start_frame``.
    fps : int, optional
        frames per second of the video, by default 30.
    """
    return save_ar_video(drift_correction_step=0, *args, **kwargs)
