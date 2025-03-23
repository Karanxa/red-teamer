from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="redteamer",
    version="0.1.0",
    author="Red Teaming Framework Team",
    author_email="example@example.com",
    description="A comprehensive framework for LLM security evaluation and benchmarking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/red-teamer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.5.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "openai>=1.2.0",
        "anthropic>=0.5.0",
        "markdown>=3.4.0",
        "jinja2>=3.1.0",
        "pydantic>=2.0.0",
        "jsonschema>=4.0.0",
        "numpy>=1.22.0",
        "scikit-learn>=1.0.0",
        "fpdf2>=2.7.0"
    ],
    entry_points={
        "console_scripts": [
            "redteamer=redteamer.cli:app",
        ],
    },
) 