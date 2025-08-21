# cstbioinfo - Custom Bioinformatics Tools

Working in bioinformatics often means that you need to quickly do some analysis, work with databases and visualize the results.

Often its not hard to do, but sometimes it needs a lot of boilerplate to get where you want to be.

This repo is my attempt at standardizing some common workflows I use. I will do by best to version the code and use tests to ensure that the code works as expected, but please be aware that this is a personal toolkit and changes may occur with minimal versioning.

## Features

- **MSA**: Multiple sequence alignment with visualization
- **Taxonomy**: Tax ID resolution and lineage analysis  
- **Embeddings**: ESM2 and ANARCII protein embeddings *(optional)*

## Installation

```bash
# Basic (MSA + taxonomy)
pip install cstbioinfo

# With embeddings
pip install "cstbioinfo[embedding]"

# Everything
pip install "cstbioinfo[full]"

# Development
pip install "cstbioinfo[dev]"
```
