#!/usr/bin/env python

import typing as tp

from setuptools import find_packages, setup

version_file = 'neudc/version.py'
readme_file = 'README.md'
requirement_file = 'requirements.txt'


def get_readme(readme_file: str) -> str:
    with open(readme_file, encoding='utf-8') as f:
        content = f.read()
    return content


def get_version(version_file: str) -> str:
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(
    fname: str = 'requirements.txt',
    with_version: bool = True,
) -> tp.List[str]:
    '''Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    '''
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line: str):
        """Parse information from a line in a requirements text file."""
        if line.startswith(('-r ', '--requirement')):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith(('-e ', '--editable ')):
                info['package'] = line.split('#egg=')[1]
            elif line.startswith((
                    '-f ',
                    '--find-links ',
                    '--extra-index-url ',
                    '-i ',
                    '--index-url ',
            )):
                pass
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # noqa
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath: str):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                if 'package' not in info:
                    continue
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='neudc',
        version=get_version(version_file),
        description=('Framework to collect dataset'
                     ' in COCO format for images/videos'
                     ' using pretrained neural networks'),
        long_description=get_readme(readme_file),
        long_description_content_type='text/markdown',
        author='MSB tech',
        author_email='ilya.basharov@gmail.com',
        keywords=[
            'computer vision',
            'dataset creation',
            'neural networks',
        ],
        python_requires='>=3.6',
        url='https://github.com/msb-tech/dataset-collection.git',
        packages=find_packages(exclude=('configs', 'tools')),
        include_package_data=True,
        classifiers=[
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
        ],
        license='MIT License',
        install_requires=parse_requirements(requirement_file),
        extras_require={
            'preinstall': parse_requirements('requirements/preinstall.txt'),
            'install': parse_requirements('requirements/install.txt'),
        },
        ext_modules=[],
        zip_safe=False,
    )
