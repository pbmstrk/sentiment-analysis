import os

from setuptools import setup, find_packages

import text_classification

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


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
