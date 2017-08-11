import os


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
