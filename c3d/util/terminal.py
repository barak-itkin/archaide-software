import numpy as np
import shutil


def configure_numpy_print():
    np_line_width = np.get_printoptions()['linewidth']
    line_width = shutil.get_terminal_size((50, np_line_width))[0]
    np.set_printoptions(precision=3, linewidth=line_width)
