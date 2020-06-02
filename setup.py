import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reclibwh", # Replace with your own username
    version="0.0.1",
    author="Ong Wai Hong",
    author_email="wai.ong11@alumni.imprtial.ac.uk",
    description="Various implementations of recommender systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whong92/recommender.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.10.0',
        'scipy>=0.19.0',
        'tqdm>=4.0.0'
        'tensorflow>=2.0.0',
        'keras>=2.3.1',
        'matplotlib>=3.0.0',
        'scikit-learn>=0.19.0',
        'pandas>=0.23.0',
        'jinja2>=2.11.2',
        'parse>=1.15.0',
        'flask>=1.1.1',
        'Flask-Script>=2.0.6',
        'Flask-Cors>=3.0.8'
    ],
    python_requires='>=3.6',
    include_package_data=True
)