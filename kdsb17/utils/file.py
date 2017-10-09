import os
from collections import OrderedDict


def read_labels(path, header=True):
    """Makes dictionary of patient_id:labels from csv file.
    Args:
        path: path to csv file containing the patient_id labels.
        header (bool): csv file has a header (True) or not (False).

    Returns:
         Dictionary of patient_id:label.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    if header:
        lines.pop(0)

    lines = [line.rstrip().split(',')[0:2] for line in lines]  # Take only the first two columns, 'id' and 'cancer'

    labels = OrderedDict((patient_id, int(label)) for patient_id, label in lines)

    return labels


def write_labels(labels, path, header=True):
    with open(path, 'w+') as f:
        if header:
            f.writelines(['id,cancer\n'])
        f.writelines([','.join([pid, str(cancer)]) + '\n' for pid, cancer in labels.items()])


def makedir(dir_path):
    """Make directory in specified path.
    Args:
        dir_path (str): directory path.
    Returns:
        Directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def make_symlinks(file_names, src_path, dst_path):
    """Make symbolic links.
    Args:
        file_names: list of file names.
        src_path: path containing the source files.
        dst_path: path where the symlinks will be saved.
    """
    for file in file_names:
        os.symlink(os.path.join(src_path, file),
                   os.path.join(dst_path, file))
