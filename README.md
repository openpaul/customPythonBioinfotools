# cstbioinfo - Custom Bioinformatics Tools

Working in bioinformatics often means that you need to quickly do some analysis, work with databases and visualize the results.

Often its not hard to do, but sometimes it needs a lot of boilerplate to get where you want to be.

This repo is my attempt at standardizing some common workflows I use. I will do by best to version the code and use tests to ensure that the code works as expected, but please be aware that this is a personal toolkit and changes may occur with minimal versioning.

## Documentation

There is some auto generated documentation available at [openpaul.github.io/customPythonBioinfotools](https://openpaul.github.io/customPythonBioinfotools/).

## Installation

```bash
git clone https://github.com/openpaul/customPythonBioinfotools
cd customPythonBioinfotools
# Basic (MSA + taxonomy)
pip install .

# With embeddings
pip install ".[embeddings]"

# Everything
pip install ".[full]"

# Development
pip install ".[dev]"
```
