# Taxonomy Module

The taxonomy module provides tools for working with taxonomic data from NCBI and UniProt databases.

## Classes

::: cstbioinfo.tax.TaxId

::: cstbioinfo.tax.TaxEntry

::: cstbioinfo.tax.UniProtTax

## Usage Examples

### Basic NCBI Taxonomy Lookup

```python
from cstbioinfo.tax import TaxId

# Create a TaxId object for humans
human_tax = TaxId(9606)

# Access taxonomic ranks
print(f"Species: {human_tax.Species}")       # Homo sapiens
print(f"Genus: {human_tax.Genus}")           # Homo  
print(f"Family: {human_tax.Family}")         # Hominidae
print(f"Order: {human_tax.Order}")           # Primates
print(f"Class: {human_tax.Class}")           # Mammalia
print(f"Phylum: {human_tax.Phylum}")         # Chordata

# Access full lineage dictionary
print(human_tax.Lineage)
# {'superkingdom': 'Eukaryota', 'kingdom': 'Metazoa', ...}
```

### UniProt Taxonomy Database

```python
from cstbioinfo.tax import UniProtTax

# Initialize UniProt taxonomy database
# This downloads and caches the database on first use
uniprot_tax = UniProtTax()

# Get taxonomic lineage
lineage = uniprot_tax.get_lineage(9606)
for entry in lineage:
    print(f"{entry.rank}: {entry.scientific_name}")

# Get rank-based lineage as dictionary
rank_lineage = uniprot_tax.get_rank_lineage(9606)
print(rank_lineage)

# Convert between names and tax IDs
tax_id = uniprot_tax.name_to_taxid("Homo sapiens")
print(tax_id)  # 9606

scientific_name = uniprot_tax.taxid_to_name(9606)
print(scientific_name)  # Homo sapiens

common_name = uniprot_tax.taxid_to_name(9606, common_name=True)
print(common_name)  # Human
```

### Working with TaxEntry Objects

```python
from cstbioinfo.tax import UniProtTax

uniprot_tax = UniProtTax()
lineage = uniprot_tax.get_lineage(9606)

# TaxEntry objects have convenient attributes
for entry in lineage:
    print(f"Tax ID: {entry.taxon_id}")
    print(f"Scientific name: {entry.scientific_name}")
    print(f"Common name: {entry.common_name}")
    print(f"Rank: {entry.rank}")
    print("---")
```

## Database Caching

The UniProtTax class automatically downloads and caches the UniProt taxonomy database locally:

- **Location**: User data directory (OS-specific)
- **Format**: SQLite database
- **Size**: ~500MB (compressed download)
- **Updates**: Use `UniProtTax(redownload=True)` to refresh

## Performance Notes

- NCBI taxonomy (TaxId) queries are fast but require internet connection
- UniProt taxonomy database is cached locally for offline use
- Database queries are indexed for optimal performance
- Lineage traversal uses recursive SQL queries
