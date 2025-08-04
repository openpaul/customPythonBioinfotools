from dataclasses import dataclass, field
from ete3 import NCBITaxa

ncbi = NCBITaxa()


@dataclass
class TaxId:
    taxid: int
    lineage: list[str] = field(init=False)
    Lineage: dict[str, str] = field(init=False)

    def __post_init__(self):
        self.lineage = ncbi.get_lineage(self.taxid)
        # Create a dictionary with ranks as keys and taxids as values

        names = ncbi.get_taxid_translator(self.lineage)
        rank = ncbi.get_rank(self.lineage)
        self.Lineage = {
            rank[taxid]: names[taxid]
            for taxid in self.lineage
            if taxid in names and taxid in rank
        }

    @property
    def Phylum(self) -> str:
        """Return the Phylum of the TaxId."""
        try:
            return self.Lineage["phylum"]
        except KeyError:
            return "Unknown"

    @property
    def Class(self) -> str:
        """Return the Class of the TaxId."""
        try:
            return self.Lineage["class"]
        except KeyError:
            return "Unknown"

    @property
    def Order(self) -> str:
        """Return the Order of the TaxId."""
        try:
            return self.Lineage["order"]
        except KeyError:
            return "Unknown"

    @property
    def Family(self) -> str:
        """Return the Family of the TaxId."""
        try:
            return self.Lineage["family"]
        except KeyError:
            return "Unknown"

    @property
    def Genus(self) -> str:
        """Return the Genus of the TaxId."""
        try:
            return self.Lineage["genus"]
        except KeyError:
            return "Unknown"

    @property
    def Species(self) -> str:
        """Return the Species of the TaxId."""
        try:
            return self.Lineage["species"]
        except KeyError:
            return "Unknown"
