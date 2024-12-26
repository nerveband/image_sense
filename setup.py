from setuptools import setup, find_packages

setup(
    name="image_sense",
    version="1.0.0",
    description="AI-powered image analysis and metadata management tool",
    author="Nerveband",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.3.0",
        "Pillow>=10.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "lxml>=4.9.0",
        "absl-py>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "image_sense=src.cli:cli",
        ],
    },
    python_requires=">=3.8",
)
