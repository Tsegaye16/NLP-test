from setuptools import setup, find_packages

setup(
    name="historical-text-analysis",
    version="1.0.0",
    description="Historical Text Analysis System using NLP Pipeline",
    author="Historical Analysis Team",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'historical-analyze=main:main',
        ],
    },
    python_requires='>=3.8',
)