import glob
import os.path as osp
import typing as tp


def find_files(file_or_dir: str,
               extensions: tp.List[str],
               recursive: bool = True) -> tp.List[str]:
    '''
    Funstion to find files with specific extensions.
    Args:
        file_or_dir (str): where to find
        extensions (list[str]): extensions for files
        recursive (bool): should work recursively or not
    Returns:
        (list[str]): files
    '''

    assert isinstance(extensions, list), \
        f'Should be list, got {type(extensions)}.'

    files = []

    if osp.isdir(file_or_dir):
        for extension in extensions:
            pattern = osp.join(file_or_dir, '**', f'*.{extension}')
            for file_path in glob.glob(pathname=pattern, recursive=recursive):
                files.append(file_path)
    elif osp.isfile(file_or_dir) and file_or_dir[file_or_dir.rfind('.') +
                                                 1:].lower() in extensions:
        files.append(file_or_dir)

    return files
