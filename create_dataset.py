import tensorflow as tf
import numpy as np
import os
import re

from dotmap import DotMap as dm

from example_bvh import EXAMPLE_BVH, EXAMPLE_HAND_BVH

def extract_bvh_file(file_content):
    """
    Extract data out of a BVH file into numpy array

    >>> extract_bvh_file(EXAMPLE_BVH).n_frames
    5
    >>> extract_bvh_file(EXAMPLE_BVH).frame_time_s
    0.008333334
    >>> extract_bvh_file(EXAMPLE_BVH).data[0, 0]
    1.528779
    >>> extract_bvh_file(EXAMPLE_BVH).data[-1, -1]
    -0.014771
    """
    lines = file_content.split('\n')
    lines = iter(lines)
    while True:
        if re.match('MOTION', next(lines)):
            break
    
    n_frames = int(re.match('Frames: (.+)', next(lines).strip()).group(1))
    frame_time_s = float(re.match('Frame Time: (.+)', next(lines).strip()).group(1))
    
    nums = []
    for i in range(n_frames):
        nums.append(list(map(float, next(lines).strip().split(' '))))

    return dm({
        "n_frames": n_frames,
        "frame_time_s": frame_time_s,
        "data": np.array(nums),
    })

def bvh_is_hand(file_content):
    """
    Ensure a BVH file contains a hand skeleton.

    >>> bvh_is_hand(EXAMPLE_BVH)
    False

    >>> bvh_is_hand(EXAMPLE_HAND_BVH)
    True

    """
    return bool(
        re.search('WRIST', file_content)
        and re.search('THUMB_CMC_FE', file_content)
        and re.search('THUMB_CMC_AA', file_content)
        and re.search('THUMB_MCP_FE', file_content)
        and re.search('THUMB_IP_FE', file_content)
        and re.search('INDEX_MCP', file_content)
        and re.search('INDEX_PIP_FE', file_content)
        and re.search('INDEX_DIP_FE', file_content)
        and re.search('MIDDLE_MCP', file_content)
        and re.search('MIDDLE_PIP_FE', file_content)
        and re.search('MIDDLE_DIP_FE', file_content)
        and re.search('RING_MCP', file_content)
        and re.search('RING_PIP_FE', file_content)
        and re.search('RING_DIP_FE', file_content)
        and re.search('PINKY_MCP', file_content)
        and re.search('PINKY_PIP_FE', file_content)
        and re.search('PINKY_DIP_FE', file_content)
    )

"""
The columns that we are extracting to predict hand motion on.
"""
USED_COLUMNS = [
    3, 4, 5, # Wrist

    8,  # THUMB_CMC_FE  Z-rot
    11, # THUMB_CMC_AA  Z-rot
    14, # THUMB_MCP_FE  Z-rot
    17, # THUMB_IP_FE   Z-rot

    18, # INDEX_MCP     X-rot
    20, # INDEX_MCP     Z-rot
    23, # INDEX_PIP_FE  Z-rot
    26, # INDEX_DIP_FE  Z-rot

    27, # MIDDLE_MCP    X-rot
    29, # MIDDLE_MCP    Z-rot
    32, # MIDDLE_PIP_FE Z-rot
    35, # MIDDLE_DIP_FE Z-rot

    36, # RING_MCP      X-rot
    38, # RING_MCP      Z-rot
    41, # RING_PIP_FE   Z-rot
    44, # RING_DIP_FE   Z-rot

    45, # PINKY_MCP     X-rot
    47, # PINKY_MCP     Z-rot
    50, # PINKY_PIP_FE  Z-rot
    53, # PINKY_DIP_FE  Z-rot
]

def extract_useful_columns(bvh_data):
    """
    Extract the degrees of freedom useful for my experiments.

    >>> extract_bvh_file(EXAMPLE_HAND_BVH).data.shape
    (1, 54)
    >>> extract_useful_columns(extract_bvh_file(EXAMPLE_HAND_BVH).data).shape
    (1, 23)
    >>> extract_useful_columns(extract_bvh_file(EXAMPLE_HAND_BVH).data)[0, 0]
    36.32338
    >>> extract_useful_columns(extract_bvh_file(EXAMPLE_HAND_BVH).data)[0, 2]
    24.17694
    >>> extract_useful_columns(extract_bvh_file(EXAMPLE_HAND_BVH).data)[0, 4]
    1.964441
    """
    return bvh_data[:, USED_COLUMNS]




if __name__ == "__main__":
    import doctest
    doctest.testmod()
