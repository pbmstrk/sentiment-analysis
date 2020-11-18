from setuptools import setup, find_packages


def get_install_requires():

    return ["pytorch_lightning>=1.0.3", "spacy", "nltk", "torch>=1.3"]


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
    name="text_classification",
    author="Paul Baumstark",
    packages=find_packages(exclude=("tests", "scripts")),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
