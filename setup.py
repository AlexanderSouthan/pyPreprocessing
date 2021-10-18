from setuptools import setup, find_packages

setup(
    name='pyPreprocessing',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'tqdm']
)
