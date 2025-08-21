"""
Taxonomy module for working with NCBI and UniProt taxonomic databases.

This module provides convenient interfaces to work with taxonomic data from both
NCBI and UniProt databases, allowing for easy taxonomy lookups, lineage retrieval,
and taxonomic data conversion.

Examples:
    Basic NCBI taxonomy lookup:

    ```python
    from cstbioinfo.tax import TaxId

    # Get taxonomic information for humans
    human = TaxId(9606)
    print(f"Species: {human.Species}")  # Homo sapiens
    print(f"Family: {human.Family}")    # Hominidae
    print(f"Order: {human.Order}")      # Primates
    ```

    UniProt taxonomy with lineage retrieval:

    ```python
    from cstbioinfo.tax import UniProtTax

    uniprot_tax = UniProtTax()
    lineage = uniprot_tax.get_lineage(9606)
    for entry in lineage:
        print(f"{entry.rank}: {entry.scientific_name}")
    ```
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
from appdirs import user_data_dir
from ete3 import NCBITaxa

ncbi = NCBITaxa()


@dataclass
class TaxId:
    """
    Interface to NCBI taxonomy database for taxonomic lookups.

    This class provides easy access to taxonomic information from NCBI's taxonomy
    database via the ETE3 library. Given a taxonomic ID, it retrieves the complete
    lineage and provides convenient properties to access different taxonomic ranks.

    Args:
        taxid: NCBI taxonomic identifier

    Attributes:
        taxid: The NCBI taxonomic ID
        lineage: List of taxonomic IDs in the lineage from root to species
        Lineage: Dictionary mapping rank names to scientific names

    Examples:
        ```python
        # Get taxonomic info for humans (taxid 9606)
        human = TaxId(9606)
        print(human.Species)  # "Homo sapiens"
        print(human.Family)   # "Hominidae"
        print(human.Order)    # "Primates"

        # Access full lineage dictionary
        print(human.Lineage)
        # {'superkingdom': 'Eukaryota', 'kingdom': 'Metazoa', ...}
        ```
    """

    taxid: int
    lineage: list[int] = field(init=False)
    Lineage: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        lineage_result = ncbi.get_lineage(self.taxid)
        self.lineage = lineage_result if lineage_result is not None else []
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


@dataclass
class TaxEntry:
    """
    Represents a single taxonomic entry with ID, names, and rank information.

    This class encapsulates taxonomic information for a single organism or
    taxonomic group, including both common and scientific names along with
    the taxonomic rank.

    Args:
        taxon_id: Taxonomic identifier
        common_name: Common name (e.g., "Human")
        scientific_name: Scientific name (e.g., "Homo sapiens")
        rank: Taxonomic rank (e.g., "species", "genus", "family")

    Examples:
        ```python
        entry = TaxEntry(9606, "Human", "Homo sapiens", "species")
        print(entry.scientific_name)  # "Homo sapiens"
        print(entry.rank)             # "species"
        ```
    """

    taxon_id: int
    common_name: str
    scientific_name: str
    rank: str

    def __str__(self) -> str:
        return f"{self.taxon_id}: {self.scientific_name} ({self.common_name}) - {self.rank}"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def _from_db_row(
        cls, row: Tuple[int, Optional[str], Optional[str], Optional[str]]
    ) -> "TaxEntry":
        """Create TaxEntry from database row tuple"""
        return cls(
            taxon_id=row[0],
            common_name=row[1] or "",  # Handle None values
            scientific_name=row[2] or "",
            rank=row[3] or "",
        )

    @classmethod
    def _from_lineage_row(
        cls, row: Tuple[int, Optional[str], Optional[str], Optional[str]]
    ) -> "TaxEntry":
        """Create TaxEntry from lineage query result (without parent field)"""
        return cls(
            taxon_id=row[0],
            common_name=row[1] or "",
            scientific_name=row[2] or "",
            rank=row[3] or "",
        )


class UniProtTax:
    """
    Interface to UniProt taxonomy database with local SQLite caching.

    This class provides access to UniProt's taxonomy database, automatically
    downloading and caching the data locally in an SQLite database for fast
    lookups. It supports lineage retrieval, name/ID conversion, and hierarchical
    taxonomy navigation.

    The database (~500MB compressed) is downloaded on first use and cached in
    the user's data directory. Updates can be forced by setting redownload=True.

    Args:
        redownload: If True, redownload and rebuild the database cache

    Examples:
        Basic usage:

        ```python
        from cstbioinfo.tax import UniProtTax

        # Initialize (downloads DB on first use)
        uniprot_tax = UniProtTax()

        # Get taxonomic lineage
        lineage = uniprot_tax.get_lineage(9606)
        for entry in lineage:
            print(f"{entry.rank}: {entry.scientific_name}")

        # Convert between names and IDs
        tax_id = uniprot_tax.name_to_taxid("Homo sapiens")  # 9606
        name = uniprot_tax.taxid_to_name(9606)              # "Homo sapiens"
        ```

        Get rank-based lineage as dictionary:

        ```python
        rank_lineage = uniprot_tax.get_rank_lineage(9606)
        # {'species': 'Homo sapiens', 'genus': 'Homo', 'family': 'Hominidae', ...}
        ```

    Note:
        Database is cached in OS-specific user data directory:
        - Linux: ~/.local/share/cstbioinfo/
        - macOS: ~/Library/Application Support/cstbioinfo/
        - Windows: %APPDATA%/openpaul/cstbioinfo/
    """

    tsv_path = "https://rest.uniprot.org/taxonomy/stream?compressed=false&fields=id%2Ccommon_name%2Cscientific_name%2Clineage%2Clinks&format=tsv&query=%28*%29"
    schema = pl.Schema(
        [
            ("Taxon Id", pl.Int64),
            ("Common name", pl.String),
            ("Scientific name", pl.String),
            ("Rank", pl.String),
            ("Parent", pl.Int64),
        ]
    )

    def __init__(self, redownload: bool = False) -> None:
        self.db_path = Path(user_data_dir("cstbioinfo", "openpaul") + "/uniprot_tax.db")
        # make user directory if it does not exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if redownload and self._db_exists():
            self.db_path.unlink()
        self.conn = self._load_or_build_db()

    def _db_exists(self) -> bool:
        try:
            with open(self.db_path, "r"):
                return True
        except FileNotFoundError:
            return False

    def _load_or_build_db(self) -> sqlite3.Connection:
        if self.db_path.exists():
            return sqlite3.connect(self.db_path.as_posix())

        print("Filling database, this might take a moment ...")
        df = pl.read_csv(
            self.tsv_path, separator="\t", schema=self.schema, ignore_errors=True
        )
        sqlite_path = f"sqlite:///{self.db_path.as_posix()}"
        df.write_database("taxonomy", sqlite_path, if_table_exists="replace")
        conn = sqlite3.connect(self.db_path.as_posix())
        self._setup_taxonomy_index(conn)
        print("Database filled and indexed.")
        return conn

    def _setup_taxonomy_index(self, conn: sqlite3.Connection) -> None:
        # create an index on the Taxon Id column for faster lookups
        query = "CREATE INDEX IF NOT EXISTS idx_taxon_id ON taxonomy(`Taxon Id`);"
        conn.execute(query)
        conn.commit()

    def get_lineage(self, tax_id: int) -> List[TaxEntry]:
        """
        Get complete taxonomic lineage as a list of TaxEntry objects.

        Retrieves the full taxonomic lineage from the given taxon ID up to the root
        of the taxonomy tree, returning each level as a TaxEntry object containing
        the taxonomic information.

        Args:
            tax_id: NCBI/UniProt taxonomic identifier

        Returns:
            List of TaxEntry objects representing the complete lineage from
            the specified taxon to the taxonomic root

        Examples:
            ```python
            lineage = uniprot_tax.get_lineage(9606)  # Human
            for entry in lineage:
                print(f"{entry.rank}: {entry.scientific_name}")
            # Output: species: Homo sapiens
            #         genus: Homo
            #         family: Hominidae
            #         ...
            ```
        """
        query = """
        WITH RECURSIVE lineage(`Taxon Id`, `Common name`, `Scientific name`, `Rank`, `Parent`) AS (
            SELECT `Taxon Id`, `Common name`, `Scientific name`, `Rank`, `Parent`
            FROM taxonomy
            WHERE `Taxon Id` = ?
            UNION ALL
            SELECT t.`Taxon Id`, t.`Common name`, t.`Scientific name`, t.`Rank`, t.`Parent`
            FROM taxonomy t
            INNER JOIN lineage l ON t.`Taxon Id` = l.`Parent`
        )
        SELECT `Taxon Id`, `Common name`, `Scientific name`, `Rank`
        FROM lineage
        """
        rows = self.conn.execute(query, (tax_id,)).fetchall()
        return [TaxEntry._from_lineage_row(row) for row in rows]

    def get_rank_lineage(self, tax_id: int) -> dict[str, str]:
        """
        Get taxonomic lineage as a rank-to-name dictionary.

        Returns the taxonomic lineage as a dictionary mapping rank names
        to scientific names, excluding non-specific "clade" entries for
        cleaner output.

        Args:
            tax_id: NCBI/UniProt taxonomic identifier

        Returns:
            Dictionary mapping rank names (e.g., 'species', 'genus') to
            scientific names (e.g., 'Homo sapiens', 'Homo')

        Examples:
            ```python
            ranks = uniprot_tax.get_rank_lineage(9606)
            print(ranks['species'])  # "Homo sapiens"
            print(ranks['family'])   # "Hominidae"
            ```
        """
        # returns a Rank: Scientifc name dict, ignores ranks that are called "clade"
        lineage = self.get_lineage(tax_id)
        return {
            entry.rank: entry.scientific_name
            for entry in lineage
            if entry.rank.lower() != "clade"
        }

    def name_to_taxid(self, name: str) -> Optional[int]:
        """
        Get taxon ID from common or scientific name.

        Searches both common name and scientific name fields to find the
        corresponding taxonomic identifier.

        Args:
            name: Common name (e.g., "Human") or scientific name (e.g., "Homo sapiens")

        Returns:
            Taxonomic ID if found, None otherwise

        Examples:
            ```python
            tax_id = uniprot_tax.name_to_taxid("Homo sapiens")  # 9606
            tax_id = uniprot_tax.name_to_taxid("Human")         # 9606
            ```
        """
        query = """
        SELECT `Taxon Id`
        FROM taxonomy
        WHERE `Common name` = ? OR `Scientific name` = ?
        """
        row = self.conn.execute(query, (name, name)).fetchone()
        if row:
            return row[0]
        return None

    def taxid_to_name(self, tax_id: int, common_name: bool = False) -> Optional[str]:
        """
        Get common or scientific name from taxon ID.

        Retrieves the name associated with a taxonomic identifier, with option
        to return either the scientific name (default) or common name.

        Args:
            tax_id: NCBI/UniProt taxonomic identifier
            common_name: If True, return common name; if False, return scientific name

        Returns:
            Requested name if found, None otherwise

        Examples:
            ```python
            scientific = uniprot_tax.taxid_to_name(9606)              # "Homo sapiens"
            common = uniprot_tax.taxid_to_name(9606, common_name=True) # "Human"
            ```
        """
        if common_name:
            query = """
            SELECT `Common name`
            FROM taxonomy
            WHERE `Taxon Id` = ?
            """
        else:
            query = """
            SELECT `Scientific name`
            FROM taxonomy
            WHERE `Taxon Id` = ?
            """
        row = self.conn.execute(query, (tax_id,)).fetchone()
        if row:
            return row[0]
        return None
