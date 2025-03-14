from setuptools import setup, find_packages

setup(
    name="DigitRecognition",
    version="0.1.0",
    packages=find_packages(),
    author="Robert Rubaszek",
    author_email="robert.rubaszek@gmail.com",
    description="An application for handwritten digits recognition",
    long_description=open("README.md").read(),
    long_description_content_type="markdown",
    url="https://github.com/rrubaszek/DigitRecognition",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)