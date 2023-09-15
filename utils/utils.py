import cv2
import numpy as np

from utils.matchers import FeatureMatcher


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
        stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    if start_frame >= stop_frame:
        raise ValueError("the starting frame must be smaller than the stopping frame.")

    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    for i in range(start_frame, stop_frame):
        if verbose: print(f"frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80), end = '\r')
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ret, frame = cap.read()
        if not ret:
            if verbose: print("Finished prematurely".ljust(80))
            break
    if verbose: print(f"frame {i-start_frame+1} of {stop_frame-start_frame}".ljust(80))
    # if verbose: print("Finished frames.")
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
    np.ndarray
        the frame corresponding to ``frame_number``. For color video,
        the channel order is RGB.
    """
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read {filename}")
    return frame


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
    np.ndarray
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


def compute_homographies(filename_src,
                         first_frame=None,
                         reference_image=None, reference_mask=None,
                         drift_correction_step=0,
                         start_frame=None, stop_frame=None,
                         min_matches_f2r=50,
                         algorithm_f2f=None,
                         algorithm_f2r=None):
    """Compute the homographies for all the frames.

    Parameters
    ----------
    filename_src : string
        name of the source video file.
    first_frame : image, optional
        first frame of the video. If not provided, it is automatically found.
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
    min_matches_f2r: int, optional
        minimum number of matches for using f2r to correct f2f drift. By default 50
    algorithm_f2r:
        feature detection algorithm used for the f2r method.
        Pass the output of cv2.SIFT_create() or similar.
    algorithm_f2f:
        feature detection algorithm used for the f2f method.
        Pass the output of cv2.SIFT_create() or similar.

    Returns
    -------
    list of np.ndarray
        list of homographies, each an array of shape (3,3).
    """
    print("Computing homographies")
    if drift_correction_step < 0:
        raise ValueError("the drift correction step must be positive.")

    first_frame = get_frame(filename_src, start_frame)
    h, w = first_frame.shape[0], first_frame.shape[1]

    if reference_image is None:
        reference_image = first_frame

    homographies = []

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
    H_f2r = first_H

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
        warped_reference_mask = cv2.warpPerspective(reference_mask, H_f2r, dsize=(w, h))
        masked_frame = np.copy(frame)
        masked_frame[np.logical_not(warped_reference_mask)] = 0

        if drift_correction_step>0 and i%drift_correction_step==0: # f2r matching
            matcher = FeatureMatcher(reference_image, masked_frame, algorithm_f2r)
            matcher.set_descriptors_1(reference_keypoints, reference_descriptors)
            matcher.find_matches()

            if len(matcher.get_matches()) >= min_matches_f2r:
                H_f2r, _ = matcher.get_homography()
                used_f2r = True

        if not used_f2r: # f2f matching
            matcher = FeatureMatcher(previous_frame, masked_frame, algorithm_f2f)
            matcher.set_descriptors_1(previous_keypoints, previous_descriptors)

            # find the homography between the previous frame and the current one
            matcher.find_matches()
            H_f2f, _ = matcher.get_homography()

            H_f2r = H_f2f@H_f2r # update the homography history

            # reset previous keypoints and frame for reuse in the next loop
            _, previous_keypoints = matcher.get_keypoints()
            _, previous_descriptors = matcher.get_descriptors()
            previous_frame = masked_frame

        homographies.append(H_f2r)
    return homographies


import time

def save_ar_video(filename_src, filename_dst, ar_layer,
                  start_frame=None, stop_frame=None,
                  ar_mask=None,
                  fps=None,
                  mark_f2r=False,
                  **kwargs):
    """Save the AR overlaid video with the F2F (frame to frame) method,
    where the reference frame can be reset after a certain number of
    frames. If the correction is done at every frame, this effectively
    becomes F2R.

    Parameters
    ----------
    filename_src : string
        name of the source video file.
    filename_dst : string
        name of the destination video file.
    ar_layer : image
        image that needs to be overlaid onto the video.
    start_frame : int, optional
        if provided, starts the rendering after this many frames.
    stop_frame : int, optional
        if provided, stops the rendering after this many frames.
        The count starts from frame 0 of the original video, and it
        needs to be greater than ``start_frame``.
    fps : int, optional
        frames per second of the video, by default the ones of the source video.
    mark_f2r: bool, optional
        mark each frame in which the homography is computed using the F2R method.
        By default False
    **kwargs:
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
    min_matches_f2r: int, optional
        minimum number of matches for using f2r to correct f2f drift. By default 50
    algorithm_f2r:
        feature detection algorithm used for the f2r method.
        Pass the output of cv2.SIFT_create() or similar.
    algorithm_f2f:
        feature detection algorithm used for the f2f method.
        Pass the output of cv2.SIFT_create() or similar.
    """
    # read the source video to get fps and resolution
    # and set the resolution of the output video as the one of the input video

    first_frame = get_frame(filename_src, start_frame)

    cap = cv2.VideoCapture(filename_src)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    t0 = time.time()
    homographies = compute_homographies(filename_src, first_frame, start_frame=start_frame, stop_frame=stop_frame, **kwargs)
    t1 = time.time()
    print(f"Time for matching: {t1-t0:.2g}s")
    print("\nWriting video")

    # create the VideoWriter object
    # and set the resolution of the output video as the one of the input video
    h, w = first_frame.shape[0], first_frame.shape[1]
    out = cv2.VideoWriter(filename=filename_dst,
                          fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                          fps=fps,
                          frameSize=(w, h))

    t0 = time.time()
    for i, (frame, homography) in enumerate(zip(frame_generator(filename_src, start_frame, stop_frame), homographies)):
        # overlay the frame with the ar layer
        ar_frame = overlay_ar(frame, homography, ar_layer, ar_mask)

        if 'drift_correction_step' in kwargs.keys() and mark_f2r:
            s = kwargs['drift_correction_step']
            if s>0 and i%s == 0:
                ar_frame = cv2.putText(ar_frame, 'F2R', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255))

        out.write(cv2.cvtColor(ar_frame, cv2.COLOR_RGB2BGR))
    out.release()
    t1 = time.time()
    print(f"Time for writing video: {t1-t0:.2g}s")


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
