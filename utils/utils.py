import cv2
import numpy as np

from utils.matchers import FeatureMatcher


class VideoFrameIndexError(IndexError):
    """Raised when the wanted video frame is out of bounds."""
    def __init__(self, message="frame number out of bounds. The video has less frames."):
        super().__init__(message)


def frame_generator(filename, start_frame=None, stop_frame=None, verbose=True):
    """Generator that yields frames from a video.

    Parameters
    ----------
    filename : string
        name of the video file.
    start_frame : int, optional
        starting frame from which to read the video, by default 0
    stop_frame : int, optional
        final frame from which to read the video, by default the final frame
    verbose : bool, optional
        by default True

    Yields
    ------
    array
        the current video frame. The channel order is RGB.

    Raises
    ------
    FileNotFoundError
        if the video file does not exist
    ValueError
        if start_frame >= stop_frame
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise FileNotFoundError(f'Video file {filename} not found!')

    if start_frame is None:
        start_frame = 0

    if stop_frame is None:
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    for i in range(start_frame, stop_frame):
        if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret, frame = cap.read()
        if not ret:
            if verbose: print("Finished prematurely".ljust(80))
            break
    if verbose: print(f"writing frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    if verbose: print("Finished frames.")
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


def save_ar_video(filename_src, filename_dst, ar_layer,
                  ar_mask=None,
                  reference_image=None, reference_mask = None,
                  drift_correction_step=0,
                  start_frame=None, stop_frame=None, fps=None,
                  min_matches_f2r=50,
                  algorithm_f2f=None,
                  algorithm_f2r=None,
                  mark_f2r=False):
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
    reference_mask : image, optional
        mask that isolates the object of interest in the reference
        image.
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
        frames per second of the video, by default the ones of the source video.
    min_matches_f2r: int, optional
        minimum number of matches for using f2r to correct f2f drift. By default 50
    """
    # read the source video to get fps and resolution
    # and set the resolution of the output video as the one of the input video
    cap = cv2.VideoCapture(filename_src)
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to video in {filename_src}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if reference_image is None:
        reference_image = first_frame
    if drift_correction_step < 0:
        raise ValueError("the drift correction step must be positive.")

    # create the VideoWriter object
    # and set the resolution of the output video as the one of the input video
    h, w = first_frame.shape[0], first_frame.shape[1]
    out = cv2.VideoWriter(filename=filename_dst,
                          fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                          fps=fps,
                          frameSize=(w, h))

    if reference_mask is None:
        reference_mask = np.ones(reference_image.shape[:2], dtype=np.uint8)*255
    # mask reference image
    reference_image[np.logical_not(reference_mask)] = 0

    # instantiate a FeatureMatcher for the reference image and the first frame
    matcher = FeatureMatcher(reference_image, first_frame, algorithm_f2r)
    matcher.find_matches()

    # get keypoints and descriptors for the first video frame
    # this is done to save computation time later
    reference_keypoints, first_keypoints = matcher.get_keypoints()
    reference_descriptors, first_descriptors = matcher.get_descriptors()

    # compute the initial homography between the reference image and the first video frame
    first_H, _ = matcher.get_homography()
    H_history = first_H

    # setup the process for f2f
    previous_frame = first_frame
    if type(algorithm_f2f) is type(algorithm_f2r):
        previous_keypoints = first_keypoints
        previous_descriptors = first_descriptors
    else:
        matcher = FeatureMatcher(first_frame, first_frame, algorithm_f2f)
        matcher._find_descriptors_1()
        previous_keypoints, _ = matcher.get_keypoints()
        previous_descriptors, _ = matcher.get_descriptors()

    for i, frame in enumerate(frame_generator(filename_src, start_frame, stop_frame)):
        used_f2r = False
        # warp the reference object mask and mask the frame
        warped_reference_mask = cv2.warpPerspective(reference_mask, H_history, dsize=(w, h))
        masked_frame = np.copy(frame)
        masked_frame[np.logical_not(warped_reference_mask)] = 0

        if drift_correction_step>0 and i%drift_correction_step==0: # f2r matching
            matcher = FeatureMatcher(reference_image, masked_frame, algorithm_f2r)
            matcher.set_descriptors_1(reference_keypoints, reference_descriptors)
            matcher.find_matches()

            if len(matcher.get_matches()) >= min_matches_f2r or drift_correction_step==0:
                H_history, _ = matcher.get_homography()
                used_f2r = True

        if not used_f2r: # f2f matching
            matcher = FeatureMatcher(previous_frame, masked_frame, algorithm_f2f)
            matcher.set_descriptors_1(previous_keypoints, previous_descriptors)

            # find the homography between the previous frame and the current one
            matcher.find_matches()
            H_f2f, _ = matcher.get_homography()

            H_history = H_f2f@H_history # update the homography history

            # reset previous keypoints and frame for reuse in the next loop
            _, previous_keypoints = matcher.get_keypoints()
            _, previous_descriptors = matcher.get_descriptors()
            previous_frame = masked_frame

        # overlay the frame with the ar layer
        ar_frame = overlay_ar(frame, H_history, ar_layer, ar_mask)

        if mark_f2r and used_f2r:
            ar_frame = cv2.putText(ar_frame, 'F2R', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255))

        out.write(cv2.cvtColor(ar_frame, cv2.COLOR_RGB2BGR))
    out.release()


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
    reference_mask : image, optional
        mask that isolates the object of interest in the reference
        image.
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
    reference_mask : image, optional
        mask that isolates the object of interest in the reference
        image.
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
