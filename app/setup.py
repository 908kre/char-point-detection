from setuptools import setup, find_packages

setup(
    name="app",
    version="0.0.0",
    description="TODO",
    author="Xinyuan Yao",
    author_email="yao.ntno@google.com",
    license="TODO",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "mlboard_client",
        "scikit-learn",
        "cytoolz",
        "pyyaml",
        "lightgbm",
        "dask",
        "joblib",
    ],
    extras_require={"dev": ["mypy", "pytest", "black",]},
    entry_points={
        "console_scripts": [
            "pipeline=app.cmd:pipeline",
            "kfold=app.cmd:kfold",
            "train=app.cmd:train",
            "dea=app.cmd:dea",
        ],
    },
)
