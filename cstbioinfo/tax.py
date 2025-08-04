from dataclasses import dataclass, field
from ete3 import NCBITaxa
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union
import polars as pl
import sqlite3
from appdirs import user_data_dir

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


@dataclass
class TaxEntry:
    taxon_id: int
    common_name: str
    scientific_name: str
    rank: str

    def __str__(self):
        return f"{self.taxon_id}: {self.scientific_name} ({self.common_name}) - {self.rank}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_db_row(cls, row: Tuple) -> "TaxEntry":
        """Create TaxEntry from database row tuple"""
        return cls(
            taxon_id=row[0],
            common_name=row[1] or "",  # Handle None values
            scientific_name=row[2] or "",
            rank=row[3] or "",
        )

    @classmethod
    def from_lineage_row(cls, row: Tuple) -> "TaxEntry":
        """Create TaxEntry from lineage query result (without parent field)"""
        return cls(
            taxon_id=row[0],
            common_name=row[1] or "",
            scientific_name=row[2] or "",
            rank=row[3] or "",
        )


class UniProtTax:
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

    def __init__(self, redownload: bool = False):
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

    def describe_tables(self) -> None:
        # print out table names and their columns
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = self.conn.execute(query).fetchall()
        for table in tables:
            print(f"Table: {table[0]}")
            query = f"PRAGMA table_info({table[0]});"
            columns = self.conn.execute(query).fetchall()
            for column in columns:
                print(f"  Column: {column[1]} - Type: {column[2]}")

    def get_lineage(self, tax_id: int) -> List[TaxEntry]:
        """Get taxonomic lineage as a list of TaxEntry objects"""
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
        return [TaxEntry.from_lineage_row(row) for row in rows]

    def get_rank_lineage(self, tax_id: int) -> dict[str, str]:
        # returns a Rank: Scientifc name dict, ignores ranks that are called "clade"
        lineage = self.get_lineage(tax_id)
        return {
            entry.rank: entry.scientific_name
            for entry in lineage
            if entry.rank.lower() != "clade"
        }

    def name_to_taxid(self, name: str) -> Optional[int]:
        """Get taxon ID from common or scientific name"""
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
        """Get common or scientific name from taxon ID"""
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
