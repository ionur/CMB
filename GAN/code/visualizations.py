# ##################################################################### #
# Authored by:
# Peikai Li
# Ipek Ilayda Onur
# ##################################################################### #

import numpy as np
from utils import Plot_CMB_Map
from pipeline import loadDataset

if __name__ == '__main__':
    filename = '../datasets/CMB_Lens_Rec_v/noise_2n/varying_ps_dataset_2n.npz'
    maps     = loadDataset(opt, filename)

    ## variables to set up the size of the map
    N = int(2 ** 7)  # this is the number of pixels in a linear dimension
    pix_size = 2.34375  # size of a pixel in arcminutes

    ## variables to set up the map plots
    X_width = N * pix_size / 60  # horizontal map width in degrees
    Y_width = N * pix_size / 60  # vertical map width in degrees

    temp_map = maps.test_label_maps[200][0]
    c_min = -max(-np.min(temp_map), np.max(temp_map))  # minimum for color bar
    c_max = +max(-np.min(temp_map), np.max(temp_map))  # maximum for color bar
    p = Plot_CMB_Map(temp_map, c_min, c_max, X_width, Y_width)

    temp_map = maps.test_target_maps[200]
    c_min = -max(-np.min(temp_map), np.max(temp_map))  # minimum for color bar
    c_max = +max(-np.min(temp_map), np.max(temp_map))  # maximum for color bar
    p = Plot_CMB_Map(temp_map, c_min, c_max, X_width, Y_width)