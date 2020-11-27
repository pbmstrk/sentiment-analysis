import os

from setuptools import find_packages, setup

import text_classification

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, file_name="requirements.txt"):
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    return lines


def get_extras_require():

    requirements = {
        "all": [
            "pytest",
            "hydra-core",
            "coverage",
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme",
        ],
        "docs": ["sphinx", "sphinx-autodoc-typehints", "sphinx-rtd-theme"],
        "scripts": ["hydra-core"],
        "tests": ["pytest", "coverage"],
    }

    return requirements


setup(
    name="text-classification",
    version=text_classification.__version__,
    author=text_classification.__author__,
    packages=find_packages(exclude=("tests", "scripts")),
    install_requires=load_requirements(),
    extras_require=get_extras_require(),
)
