from setuptools import setup

install_requires = [
    'joblib==1.2.0',
    'matplotlib==3.5.2',
    'mmcv==1.4.5',
    'munch==2.5.0',
    'numpy==1.24.3',
    'pandas==1.5.3',
    'PyYAML==6.0.1',
    'rdkit==2022.9.1',
    'rdkit_pypi==2021.9.4',
    'ruamel.yaml==0.17.21',
    'scikit_learn==1.2.2',
    'setuptools==59.5.0',
    'tqdm==4.65.0',
    'typed-argument-parser==1.7.2',
]

setup(
    name='GESS',
    version='0.1',
    author="Derek Zou",
    description='GeSS: Benchmarking Geometric Deep Learning under Scientific Applications with Distribution Shifts',
    install_requires=install_requires,
    package_dir={"GESS": "GESS"},
    entry_points={
        'console_scripts': [
            'gess-run = GESS.core.main:main',
        ],
    },
    python_requires=">=3.8",
)
