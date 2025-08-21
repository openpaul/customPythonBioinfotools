# MSA Module

Multiple sequence alignment tools with visualization capabilities.

This requires that you have MAFFT installed on your system. You can install it via conda.


::: cstbioinfo.msa


## Usage Examples


### Get an MSA
```python
from cstbioinfo.msa import msa

sequences = [
    "MKTAYIAKQRQISFVKSHFSR",
    "MKTAYIAKQRQISFVKSHFSR",
    "MKTAYIAKQRQISFVKSHFSR",
]
msa = msa(sequences)
```


### Plot an MSA
```python
from cstbioinfo.msa import plot_msa

# Does not need to be an MSA object, just a list of sequences
mv = plot_msa(sequences, title="Example MSA")
# Display the plot properly (semicolon suppresses output)
mv.plotfig();
```

