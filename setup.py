import setuptools

from skquery import __version__


def readme():
    with open("README.md", "r") as f:
        return f.read()


def requirements():
    with open("requirements.txt", "r") as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


setuptools.setup(
    name="scikit-query",
    version=__version__,
    description="scikit-query is a Python library "
    "for active query strategies in constrained clustering on top of SciPy and "
    "scikit-learn.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "active clustering",
        "semi-supervised clustering",
        "constrained clustering",
        "pattern recognition",
        "machine learning",
        "artificial intelligence",
    ],
    url="https://github.com/aymericb213/scikit-query",
    author="Aymeric Beauchamp",
    python_requires=">=3.10",
    author_email="aymeric.beauchamp@univ-orleans.fr",
    license="BSD 3-Clause License",
    packages=setuptools.find_packages(),
    install_requires=requirements(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    zip_safe=False,
)