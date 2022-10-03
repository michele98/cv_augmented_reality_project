import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.utils import frame_generator


def display_frames(filename, starting_frame=0, refresh_output=True):
    """Display video frames from a video file, starting from a
    starting frame.

    Parameters
    ----------
    filename : string
        name of the video file
    starting_frame : int, optional
        by default 0
    refresh_output : bool, optional
        choose whether each new frame clears the previous one, by
        default True
    """
    def display():
        if refresh_output: clear_output(wait = True)
        plt.imshow(frame)
        plt.gca().set_axis_off()
        plt.tight_layout(pad=0)
        plt.show()
        print(f"frame {i}")

    for i, frame in enumerate(frame_generator(filename)):
        try:
            if i >= starting_frame:
                display()
        except KeyboardInterrupt:
            display()
            break

    if i < starting_frame:
        print('There are less frames than starting_frame!')
    print('Capture released')
