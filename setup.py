from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

setup(
    name="digit_recognition_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
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