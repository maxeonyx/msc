import re
import pickle

from dotmap import DotMap as dm

from mx.prelude import *

from ._bvh_templates import TEMPLATE_BVH, TEMPLATE_RIGHT_HAND_BVH, TEMPLATE_LEFT_HAND_BVH

__all__, export = exporter()

# assumes running with working directory as root of the repo
DEFAULT_BVH_DIR = Path(__file__).parent / "BVH"
DEFAULT_OUTPUT_BVH_DIR = Path.cwd() / "_anims"
export("DEFAULT_BVH_DIR")
export("DEFAULT_OUTPUT_BVH_DIR")

def extract_bvh_file(file_content):
    """
    Extract data out of a BVH file into numpy array

    >>> extract_bvh_file(TEMPLATE_BVH).n_frames
    5
    >>> extract_bvh_file(TEMPLATE_BVH).frame_time_s
    0.008333334
    >>> extract_bvh_file(TEMPLATE_BVH).data[0, 0]
    1.528779
    >>> extract_bvh_file(TEMPLATE_BVH).data[-1, -1]
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
        "data": np.array(nums, dtype=np.float32),
    })

def bvh_is_hand(file_content):
    """
    Ensure a BVH file contains a hand skeleton.

    >>> bvh_is_hand(TEMPLATE_BVH)
    False

    >>> bvh_is_hand(TEMPLATE_LEFT_HAND_BVH)
    True

    >>> bvh_is_hand(TEMPLATE_RIGHT_HAND_BVH)
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

Includes all DOF from all joints, but not the *location* of the hand.
"""
COL_ALL_JOINTS = [
    3,  4,  5, # Wrist

    6,  7,  8,  # THUMB_CMC_FE
    9,  10, 11, # THUMB_CMC_AA
    12, 13, 14, # THUMB_MCP_FE
    15, 16, 17, # THUMB_IP_FE

    18, 19, 20, # INDEX_MCP
    21, 22, 23, # INDEX_PIP_FE
    24, 25, 26, # INDEX_DIP_FE

    27, 28, 29, # MIDDLE_MCP
    30, 31, 32, # MIDDLE_PIP_FE
    33, 34, 35, # MIDDLE_DIP_FE

    36, 37, 38, # RING_MCP
    39, 40, 41, # RING_PIP_FE
    42, 43, 44, # RING_DIP_FE

    45, 46, 47, # PINKY_MCP
    48, 49, 50, # PINKY_PIP_FE
    51, 52, 53, # PINKY_DIP_FE
]

"""
The columns that we are extracting to predict hand motion on.

Includes only the DOF that vary.
"""
COL_USEFUL = [
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

def extract_columns(bvh_data, cols):
    """
    Extract the degrees of freedom useful for my experiments.

    >>> extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data.shape
    (1, 54)
    >>> extract_columns(extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data, COL_USEFUL).shape
    (1, 23)
    >>> extract_columns(extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data, COL_ALL_JOINTS).shape
    (1, 51)
    >>> extract_columns(extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data, COL_USEFUL)[0, 0]
    36.32338
    >>> extract_columns(extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data, COL_USEFUL)[0, 2]
    24.17694
    >>> extract_columns(extract_bvh_file(TEMPLATE_LEFT_HAND_BVH).data, COL_USEFUL)[0, 4]
    1.964441
    """
    return bvh_data[:, cols]


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
            yield filepath, text

# chunk generator into batches of size n
def chunk(g, n):
    i = iter(g)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(i))
            yield chunk
        except StopIteration:
            break

def get_bvh_data(columns, bvh_dir=None, convert_deg_to_rad=True):
    """
    Get a list of all the BVH data in the manipnet dataset.

    Returns list of (filename, obj) pairs where obj.data has shape (n_frames, n_deg_of_free)

    Example below based on the number of files in the manipnet database as of the time of writing.

    >>> len(list(get_bvh_data()))
    62

    """
    for (l_fname, l_text), (r_fname, r_text) in chunk(read_hand_bvh_files(bvh_dir or DEFAULT_BVH_DIR), 2):

        if "left" in r_fname.lower() and "right" in l_fname.lower():
            l_fname, r_fname = r_fname, l_fname
            l_text, r_text = r_text, l_text

        assert "left" in l_fname.lower(), f"Filename {l_fname} does not fit the pattern."
        assert "right" in r_fname.lower(), f"Filename {l_fname} does not fit the pattern."
        assert l_fname.lower().replace("left", "right") == r_fname.lower()

        l_obj = extract_bvh_file(l_text)
        r_obj = extract_bvh_file(r_text)

        assert l_obj.data.shape == r_obj.data.shape
        assert l_obj.frame_time_s == r_obj.frame_time_s

        if columns == "useful":
            cols = COL_USEFUL
        elif columns == "all":
            cols = COL_ALL_JOINTS
        else:
            raise ValueError(f"Unknown 'columns' config value: {columns}")

        # extract a subset of the columns
        l_obj.data = extract_columns(l_obj.data, cols)
        r_obj.data = extract_columns(r_obj.data, cols)

        name = os.path.basename(os.path.normpath(os.path.os.path.commonprefix([l_fname, r_fname])))

        # axes are [frame, hand, dof]
        data = np.stack([l_obj.data, r_obj.data], axis=1)

        if convert_deg_to_rad:
            data = data / 360 * (2*np.pi)

        yield name, data, data.shape[0]

@export
def np_dataset_parallel_lists(force=False, convert_deg_to_rad=True, columns=None) -> tuple[list[str], list[np.ndarray], list[int]]:
    """
    Get a numpy dataset of all the BVH data in the manipnet dataset.

    >>> len(np_dataset_parallel_lists())
    3

    >>> len(np_dataset_parallel_lists()[0])
    62

    >>> np_dataset_parallel_lists()[0][0]
    './BVH/bottle1_body1/'

    >>> np_dataset_parallel_lists()[1][0].shape
    (8000, 2, 23)

    """

    ds_path = "./_cache/dataset.pickle"

    if not force:
        try:
            with open(ds_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError as f:
            print(f'Couldnt load saved dataset, generating a fresh dataset to "{ds_path}" ...', file=sys.stderr)
    else:
        print(f'Forcing generation of a fresh dataset to "{ds_path}" ...', file=sys.stderr)

    data = [(f, a, n) for f, a, n in get_bvh_data(convert_deg_to_rad=convert_deg_to_rad, columns=columns)]
    data.sort(key=lambda x: x[0])
    # sort by filename
    filenames, angles, n_frames = [], [], []
    for f, a, n in data:
        filenames.append(f)
        angles.append(a)
        n_frames.append(n)

    Path(ds_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ds_path, "wb") as f:
        pickle.dump((filenames, angles, n_frames), f)

    return filenames, angles, n_frames

@export
def write_bvh_files(data, name, column_map, output_dir=None, convert_rad_to_deg=True):
    """
    Write a new pair of animation files (left and right hands) from a single data array.
    """

    if tf.is_tensor(data):
        data = data.numpy()

    assert len(data.shape) == 3 or (len(data.shape) == 4 and data.shape[3] == 3), f"data must be a 3D tensor with shape (n_frames, n_hands, n_dof_per_hand) or a 4D tensor with shape (n_frames, n_hands, n_joints_per_hand, n_dof_per_joint=3), got {data.shape}"
    if len(data.shape) == 4:
        data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2]*data.shape[3]])
    assert data.shape[0] >= 0, "data must have at least one frame"
    assert data.shape[1] in [1, 2], f"data must have 1 or 2 hands, got {data.shape[1]}"
    assert data.shape[2] == 23 or data.shape[2] == 51, "data must have 23 or 51 degrees of freedom"

    if convert_rad_to_deg:
        # convert back to degrees
        data = data / (2*np.pi) * 360

    output_dir = output_dir or DEFAULT_OUTPUT_BVH_DIR

    left_hand = TEMPLATE_LEFT_HAND_BVH
    right_hand = TEMPLATE_RIGHT_HAND_BVH

    def write_file(content, filename, data):
        with open(filename, 'w') as out_file:

            lines = content.split('\n')
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
            print(f"Frames: {data.shape[0]}", file=out_file)

            # echo frame time
            line = next(lines)
            print(line, file=out_file)

            dummy_nums = list(map(float, next(lines).strip().split(' ')))
            for i in range(data.shape[0]):
                nums = [n for n in dummy_nums]
                for col in range(len(column_map)):
                    nums[column_map[col]] = data[i, col]

                print(' '.join(str(n) for n in nums), file=out_file)

    left_filename = os.path.join(output_dir, name + '.left.generated.bvh')
    write_file(left_hand, left_filename, data[:, 0])
    if data.shape[1] == 2:
        right_filename = os.path.join(output_dir, name + '.right.generated.bvh')
        write_file(right_hand, right_filename, data[:, 1])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
