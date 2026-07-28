"""Immune-focused MSA helpers based on immunum numbering.

This module provides utilities to:
- detect nucleotide vs amino-acid inputs,
- scan nucleotide sequences in six frames to find receptor-like domains,
- build an alignment anchored by IMGT/Kabat numbering positions,
- visualize the result with pymsaviz, including optional FR/CDR annotation tracks.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from immunum import Annotator
from .utils import (
    find_receptor_boundaries,
    imgt_sort,
    infer_sequence_type,
    kabat_sort,
)
from pymsaviz import MsaViz


def _raw_seq(seq: str) -> str:
    return seq.upper()


def _extract_best_aa_domain(
    seq: str,
    min_confidence: float,
    chains: Sequence[str],
    scheme: str,
    min_orf_len: int,
) -> tuple[str, dict[str, Any] | None]:
    seq_type = infer_sequence_type(seq)
    if seq_type == "amino_acid":
        return _raw_seq(seq), None

    hits = find_receptor_boundaries(
        seq,
        min_confidence=min_confidence,
        min_orf_len=min_orf_len,
        chains=chains,
        scheme=scheme,
    )
    if not hits:
        raise ValueError("No receptor-like domain found in nucleotide sequence.")
    return hits[0]["domain_seq"], hits[0]


def extract_receptor_sequence(
    sequence: str | SeqRecord,
    min_confidence: float = 0.5,
    min_orf_len: int = 20,
    chains: Sequence[str] = ("H", "K", "L", "A", "B", "G", "D"),
    scheme: str = "imgt",
) -> dict[str, Any]:
    """Compatibility wrapper.

    Canonical implementation is in ``cstbioinfo.immune.utils``.
    """
    from .utils import extract_receptor_sequence as _extract_receptor_sequence

    return _extract_receptor_sequence(
        sequence=sequence,
        min_confidence=min_confidence,
        min_orf_len=min_orf_len,
        chains=chains,
        scheme=scheme,
    )


def _region_ranges_from_numbering(
    global_positions: Sequence[str],
    target_numbering_positions: Sequence[str],
    region_lengths: Sequence[int],
) -> list[tuple[int, int] | None]:
    """Map FR/CDR region lengths to global 1-based alignment column ranges.

    This is gap-aware because ranges are projected through numbering-position
    columns of the final alignment.
    """
    col_by_pos = {pos: idx + 1 for idx, pos in enumerate(global_positions)}
    ordered_target_pos = list(target_numbering_positions)

    ranges: list[tuple[int, int] | None] = []
    cursor = 0
    for region_len in region_lengths:
        if region_len <= 0:
            ranges.append(None)
            continue

        chunk = ordered_target_pos[cursor : cursor + region_len]
        cursor += region_len
        cols = [col_by_pos[p] for p in chunk if p in col_by_pos]
        if not cols:
            ranges.append(None)
            continue

        ranges.append((min(cols), max(cols)))

    return ranges


def _aggregate_region_ranges(
    per_sequence_ranges: Sequence[Sequence[tuple[int, int] | None]],
) -> list[tuple[int, int] | None]:
    """Aggregate region ranges across sequences into global min-start/max-end ranges."""
    if not per_sequence_ranges:
        return []

    n_regions = len(per_sequence_ranges[0])
    aggregated: list[tuple[int, int] | None] = []
    for i in range(n_regions):
        starts: list[int] = []
        ends: list[int] = []
        for row in per_sequence_ranges:
            rng = row[i]
            if rng is None:
                continue
            starts.append(rng[0])
            ends.append(rng[1])
        if starts and ends:
            aggregated.append((min(starts), max(ends)))
        else:
            aggregated.append(None)
    return aggregated


def number_msa(
    sequences: Sequence[str | SeqRecord],
    seq_ids: Sequence[str] | None = None,
    scheme: str = "imgt",
    chains: Sequence[str] = ("H", "K", "L", "A", "B", "G", "D"),
    min_confidence: float = 0.5,
    min_orf_len: int = 20,
) -> tuple[list[SeqRecord], list[str], list[dict[str, Any]]]:
    """Build an immune MSA from immunum numbering positions.

    Returns aligned records, ordered numbering positions, and per-sequence
    metadata.
    """
    if not sequences:
        return [], [], []

    anno = Annotator(chains=list(chains), scheme=scheme, min_confidence=min_confidence)
    metadata: list[dict[str, Any]] = []
    numbering_maps: list[dict[str, str]] = []

    normalized_ids: list[str] = []
    for i, item in enumerate(sequences):
        if isinstance(item, SeqRecord):
            seq = str(item.seq)
            seq_id = item.id
        else:
            seq = item
            seq_id = f"seq{i + 1}"

        if seq_ids is not None:
            seq_id = seq_ids[i]
        normalized_ids.append(seq_id)

        seq_type = infer_sequence_type(seq)
        if seq_type == "nucleotide":
            hits = find_receptor_boundaries(
                seq,
                min_confidence=min_confidence,
                min_orf_len=min_orf_len,
                chains=chains,
                scheme=scheme,
                annotator=anno,
            )
            if not hits:
                raise ValueError(
                    f"No receptor-like domain found for sequence {seq_id}."
                )

            hit = hits[0]
            numbering = {
                str(k): str(v) for k, v in dict(hit.get("numbering", {}) or {}).items()
            }
            if not numbering:
                # Fallback if boundary hit has no numbering payload for any reason.
                res = anno.number(str(hit.get("orf_seq", "")))
                numbering = {
                    str(k): str(v)
                    for k, v in dict(getattr(res, "numbering", {}) or {}).items()
                }
            if not numbering:
                raise ValueError(f"Failed to number sequence {seq_id}.")

            numbering_maps.append(numbering)
            metadata.append(
                {
                    "id": seq_id,
                    "input_type": seq_type,
                    "selected_chain": hit.get("chain"),
                    "confidence": float(hit.get("confidence", 0.0) or 0.0),
                    "best_hit": hit,
                }
            )
            continue

        aa_seq = _raw_seq(seq)
        res = anno.number(aa_seq)
        if getattr(res, "chain", None) is None:
            raise ValueError(f"Failed to number sequence {seq_id}.")

        numbering = {
            str(k): str(v) for k, v in dict(getattr(res, "numbering", {}) or {}).items()
        }
        numbering_maps.append(numbering)
        metadata.append(
            {
                "id": seq_id,
                "input_type": seq_type,
                "selected_chain": getattr(res, "chain", None),
                "confidence": float(getattr(res, "confidence", 0.0) or 0.0),
                "best_hit": None,
            }
        )

    all_positions_unsorted: list[str] = []
    seen_positions: set[str] = set()
    for numbering in numbering_maps:
        for pos in numbering.keys():
            if pos not in seen_positions:
                seen_positions.add(pos)
                all_positions_unsorted.append(pos)

    scheme_l = str(scheme).lower()
    if scheme_l == "imgt":
        all_positions = imgt_sort(all_positions_unsorted)
    elif scheme_l == "kabat":
        all_positions = kabat_sort(all_positions_unsorted)
    else:
        all_positions = all_positions_unsorted

    aligned: list[SeqRecord] = []
    for seq_id, numbering in zip(normalized_ids, numbering_maps):
        aligned_str = "".join(numbering.get(pos, "-") for pos in all_positions)
        aligned.append(SeqRecord(Seq(aligned_str), id=seq_id, description=""))

    return aligned, all_positions, metadata


def plot_numbered_msa(
    sequences: Sequence[str | SeqRecord],
    seq_ids: Sequence[str] | None = None,
    scheme: str = "imgt",
    chains: Sequence[str] = ("H", "K", "L", "A", "B", "G", "D"),
    min_confidence: float = 0.5,
    min_orf_len: int = 20,
    annotate_regions_from: int = 0,
    annotate_regions_strategy: str = "all",
    annotate_regions: bool = True,
    annotate_region_colors: dict[str, str] | None = None,
    annotate_region_boundaries: bool = True,
    boundary_marker: str = "|",
    boundary_color: str = "black",
    boundary_size: float = 9.0,
    annotate_low_consensus_threshold: float | None = 50.0,
    **kwargs: Any,
) -> tuple[MsaViz, list[dict[str, Any]], list[str]]:
    """Create a pymsaviz object from numbering-based immune alignment.

    The plot can include:
    - x-markers on low-consensus columns,
    - FR/CDR text annotations from one representative sequence.
    """
    aligned, positions, metadata = number_msa(
        sequences=sequences,
        seq_ids=seq_ids,
        scheme=scheme,
        chains=chains,
        min_confidence=min_confidence,
        min_orf_len=min_orf_len,
    )

    if not aligned:
        raise ValueError("No sequences to plot.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta") as tmp:
        SeqIO.write(aligned, tmp.name, "fasta")
        mv = MsaViz(tmp.name, **kwargs)

    if annotate_low_consensus_threshold is not None:
        low_consensus_cols: list[int] = []
        ident_list = mv._get_consensus_identity_list()
        for pos, ident in enumerate(ident_list, 1):
            if ident <= annotate_low_consensus_threshold:
                low_consensus_cols.append(pos)
        if low_consensus_cols:
            mv.add_markers(low_consensus_cols, marker="x", color="blue")

    if annotate_regions:
        region_labels = ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]
        anno = Annotator(
            chains=list(chains), scheme=scheme, min_confidence=min_confidence
        )

        strategy = annotate_regions_strategy.lower()
        if strategy not in {"all", "first"}:
            raise ValueError("annotate_regions_strategy must be 'all' or 'first'.")

        per_sequence_ranges: list[list[tuple[int, int] | None]] = []

        indices: list[int]
        if strategy == "first":
            indices = [max(0, min(annotate_regions_from, len(aligned) - 1))]
        else:
            indices = list(range(len(sequences)))

        for idx in indices:
            seq_input = sequences[idx]
            seq_str = (
                str(seq_input.seq) if isinstance(seq_input, SeqRecord) else seq_input
            )
            try:
                seq_aa, _ = _extract_best_aa_domain(
                    seq_str,
                    min_confidence,
                    chains,
                    scheme,
                    min_orf_len,
                )
            except ValueError:
                continue

            seg = anno.segment(seq_aa)
            region_values = [getattr(seg, label, "") or "" for label in region_labels]
            region_lengths = [len(x) for x in region_values]

            seq_num = anno.number(seq_aa)
            seq_numbering = dict(getattr(seq_num, "numbering", {}) or {})
            seq_ranges = _region_ranges_from_numbering(
                global_positions=positions,
                target_numbering_positions=[str(k) for k in seq_numbering.keys()],
                region_lengths=region_lengths,
            )
            per_sequence_ranges.append(seq_ranges)

        if not per_sequence_ranges:
            region_ranges = [None] * len(region_labels)
        elif strategy == "first":
            region_ranges = per_sequence_ranges[0]
        else:
            region_ranges = _aggregate_region_ranges(per_sequence_ranges)

        default_region_colors = {
            "fr1": "#1b9e77",
            "cdr1": "#d95f02",
            "fr2": "#1b9e77",
            "cdr2": "#d95f02",
            "fr3": "#1b9e77",
            "cdr3": "#7570b3",
            "fr4": "#1b9e77",
        }
        color_map = default_region_colors.copy()
        if annotate_region_colors is not None:
            for key, value in annotate_region_colors.items():
                color_map[key.lower()] = value

        boundary_positions: list[int] = []
        for label, rng in zip(region_labels, region_ranges):
            if rng is None:
                continue
            color = color_map.get(label, "black")
            mv.add_text_annotation(
                rng,
                label.upper(),
                text_color=color,
                range_color=color,
            )
            boundary_positions.append(rng[0])
            boundary_positions.append(rng[1])

        if annotate_region_boundaries and boundary_positions:
            boundary_positions = sorted(set(boundary_positions))
            mv.add_markers(
                boundary_positions,
                marker=boundary_marker,
                color=boundary_color,
                size=boundary_size,
            )

    return mv, metadata, positions
