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
USEFUL_COLUMNS = [
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
    return bvh_data[:, USEFUL_COLUMNS]


def all_bvh_files(path):
    """
    Walk a directory tree and generate a list of all of the Hand BVH files in the tree.
    """

    for dirpath, subdirpaths, files in os.walk(path):
        yield from (dirpath + "/" + f for f in files if f.endswith('.bvh'))


def read_hand_bvh_files(path):
    """
    Read BVH files and yield their contents if they are hand files.
    """

    for filepath in all_bvh_files(path):
        file = open(filepath)
        text = file.read()
        if bvh_is_hand(text):
            yield(filepath, text)

def get_bvh_data():
    """
    Get a list ofof all the BVH data in the manipnet repo.
    
    Returns list of (filename, obj) pairs where obj.data has shape (n_frames, n_deg_of_free)

    Example below based on the number of files in the manipnet database as of the time of writing.

    >>> len(list(get_bvh_data()))
    124
    """
    for name, text in read_hand_bvh_files("manipnet/Data/SimpleVisualizer/Assets/BVH"):
        
        obj = extract_bvh_file(text)
        obj.data = extract_useful_columns(obj.data)
        if "right" in name:
            is_right_hand = True
        elif "left" in name:
            is_right_hand = False
        else:
            raise Exception(f"Found file {name} with no left/right distinction")
        yield name, obj.n_frames, obj.data, is_right_hand

def np_dataset():

    data = [x for x in get_bvh_data()]
    data = np.array(data, dtype=np.object_)
    # sort by filename
    data = data[data[:, 0].argsort()]
    return data

def tf_dataset(force=False):
    """
    Load the cached tensorflow dataset or create it from the BVH files on disk, and then cache it on disk.
    """

    ds_element_spec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=[None, len(USEFUL_COLUMNS)]),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    )

    ds_path = './cached_dataset'

    ds_compression = 'GZIP'
    
    # by default, try to load and return the saved dataset.
    # if force, don't try to load.
    # if loading fails, create a fresh dataset.
    if not force:
        try:
            return tf.data.experimental.load(ds_path, ds_element_spec, ds_compression)
        except FileNotFoundError as f:
            print('Couldnt load saved dataset, generating a fresh dataset to "./cached_dataset/" ...')
    else:
        print('Forcing generation of a fresh dataset to "./cached_dataset/" ...')

    d = tf.data.Dataset.from_generator(
        get_bvh_data,
        output_signature=ds_element_spec,
    ).cache()

    tf.data.experimental.save(d, ds_path, 'GZIP')

    return d


def load_one_bvh_file(filename, convert_deg_to_rad=True):
    """
    Load one BVH file and return the frame data.
    """
    with open(filename, 'r') as f:
        content = f.read()
    obj = extract_bvh_file(content)
    obj.data = extract_useful_columns(obj.data)

    if convert_deg_to_rad:
        # convert back to degrees
        obj.data = obj.data /  360 * (2*np.pi)
    return obj.data


def write_bvh_file(filename, data, convert_rad_to_deg=True):
    """
    Copy an existing BVH file and write new animation tracks into it.
    """

    if convert_rad_to_deg:
        # convert back to degrees
        data = data / (2*np.pi) * 360

    with open(filename, 'r') as f:
        file_content = f.read()
    
    with open(filename + '.generated.bvh', 'w') as out_file:
    
        lines = file_content.split('\n')
        lines = iter(lines)

        # echo skeleton definition and motion line
        while True:
            line = next(lines)
            print(line, file=out_file)
            if re.match('MOTION', line):
                break
        
        # write new frame count
        line = next(lines)
        n_frames = int(re.match('Frames: (.+)', line.strip()).group(1))
        assert data.shape[0] <= n_frames, "Writing more frames than in the base file is currently not supported."
        print(f"Frames: {data.shape[0]}", file=out_file)

        # echo frame time
        line = next(lines)
        print(line, file=out_file)
        
        nums = []
        for i in range(data.shape[0]):
            nums = list(map(float, next(lines).strip().split(' ')))

            for col in range(len(USEFUL_COLUMNS)):
                nums[USEFUL_COLUMNS[col]] = data[i, col]
            
            print(' '.join(str(n) for n in nums), file=out_file)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
