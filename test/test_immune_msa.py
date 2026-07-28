from types import SimpleNamespace

import pytest

from cstbioinfo.immune import msa as immune_msa
from cstbioinfo.immune import utils as immune_utils


class FakeAnnotator:
    def __init__(self, chains=None, scheme="imgt", **kwargs):
        self.chains = chains
        self.scheme = scheme

    def number(self, seq: str):
        # Provide deterministic pseudo-numbering over first residues.
        if not seq:
            return SimpleNamespace(chain=None, confidence=0.0, numbering={})
        numbering = {str(i + 1): aa for i, aa in enumerate(seq[:6])}
        return SimpleNamespace(
            chain="H",
            confidence=0.9,
            numbering=numbering,
            query_start=0,
            query_end=min(len(seq), 6) - 1,
        )

    def segment(self, seq: str):
        # Simple fixed segmentation for tests.
        return SimpleNamespace(
            fr1=seq[:2],
            cdr1=seq[2:3],
            fr2=seq[3:4],
            cdr2=seq[4:5],
            fr3=seq[5:6],
            cdr3=seq[6:7],
            fr4=seq[7:8],
            prefix="",
            postfix="",
        )


def test_infer_sequence_type_basic():
    assert immune_msa.infer_sequence_type("ACGTACGT") == "nucleotide"
    assert immune_msa.infer_sequence_type("EVQLVESGGGLV") == "amino_acid"


def test_find_receptor_boundaries_forward_hit():
    dna = "ATGGCTGCTGCTGCTGCTGCTGCTTAA"  # MAAAAAAA*
    hits = immune_msa.find_receptor_boundaries(
        dna_seq=dna,
        min_confidence=0.5,
        min_orf_len=2,
        annotator=FakeAnnotator(),
    )
    assert len(hits) >= 1
    best = hits[0]
    assert best["chain"] == "H"
    assert best["nt_start"] >= 0
    assert best["nt_end"] > best["nt_start"]


def test_number_msa_uses_numbering_positions(monkeypatch):
    monkeypatch.setattr(immune_msa, "Annotator", FakeAnnotator)

    aligned, positions, metadata = immune_msa.number_msa(
        sequences=["EVQLVES", "EVQLVDS"],
        seq_ids=["a", "b"],
    )

    assert positions == ["1", "2", "3", "4", "5", "6"]
    assert str(aligned[0].seq) == "EVQLVE"
    assert str(aligned[1].seq) == "EVQLVD"
    assert metadata[0]["id"] == "a"
    assert metadata[1]["id"] == "b"


def test_number_msa_from_nucleotide(monkeypatch):
    monkeypatch.setattr(immune_msa, "Annotator", FakeAnnotator)

    # Encodes MAAAAAA
    dna = "ATGGCTGCTGCTGCTGCTGCT"
    aligned, positions, metadata = immune_msa.number_msa([dna], min_orf_len=2)

    assert len(aligned) == 1
    assert len(positions) > 0
    assert metadata[0]["input_type"] == "nucleotide"


def test_region_ranges_from_numbering_gap_aware():
    global_positions = ["1", "2", "3", "4", "5", "6", "7", "8"]
    # Representative sequence is missing position 4 in its own numbering.
    target_numbering_positions = ["1", "2", "3", "5", "6", "7", "8"]
    # FR1=2 aa, CDR1=1 aa, FR2=2 aa, CDR2=1 aa, FR3=1 aa, CDR3/FR4 absent
    region_lengths = [2, 1, 2, 1, 1, 0, 0]

    ranges = immune_msa._region_ranges_from_numbering(
        global_positions=global_positions,
        target_numbering_positions=target_numbering_positions,
        region_lengths=region_lengths,
    )

    # FR2 should map across columns for numbering 5 and 6 (i.e., (5, 6)),
    # not naive sequential (4, 5).
    assert ranges[0] == (1, 2)
    assert ranges[1] == (3, 3)
    assert ranges[2] == (5, 6)
    assert ranges[3] == (7, 7)
    assert ranges[4] == (8, 8)


def test_extract_receptor_sequence_amino_acid(monkeypatch):
    monkeypatch.setattr(immune_utils, "Annotator", FakeAnnotator)

    out = immune_msa.extract_receptor_sequence("EVQLVES")
    assert out["input_type"] == "amino_acid"
    assert out["chain"] == "H"
    assert out["sequence"] == "EVQLVES"
    assert isinstance(out["numbering"], dict)


def test_extract_receptor_sequence_nucleotide(monkeypatch):
    monkeypatch.setattr(immune_utils, "Annotator", FakeAnnotator)

    dna = "ATGGCTGCTGCTGCTGCTGCT"  # MAAAAAA
    out = immune_msa.extract_receptor_sequence(dna, min_orf_len=2)
    assert out["input_type"] == "nucleotide"
    assert out["chain"] == "H"
    assert len(out["sequence"]) > 0
    assert isinstance(out["best_hit"], dict)


def test_number_msa_uses_imgt_sorting(monkeypatch):
    class FakeAnnotatorIMGT:
        def __init__(self, chains=None, scheme="imgt", min_confidence=0.5):
            self.chains = chains
            self.scheme = scheme

        def number(self, seq: str):
            return SimpleNamespace(
                chain="H",
                confidence=0.9,
                numbering={"112": "A", "111A": "B", "111": "C"},
                query_start=0,
                query_end=2,
            )

        def segment(self, seq: str):
            return SimpleNamespace(
                fr1="", cdr1="", fr2="", cdr2="", fr3="", cdr3="", fr4=""
            )

    monkeypatch.setattr(immune_msa, "Annotator", FakeAnnotatorIMGT)

    aligned, positions, _ = immune_msa.number_msa(["EVQLVES"], scheme="imgt")

    assert positions == ["111", "111A", "112"]
    assert str(aligned[0].seq) == "CBA"
