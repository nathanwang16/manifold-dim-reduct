from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import pandas as pd
from pyfaidx import Fasta

from logger import configure_logging, get_logger, log_metrics


CANONICAL_CHROMS: List[str] = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CANONICAL_CHROM_SET = set(CANONICAL_CHROMS)
TRAIN_CHROMS = {f"chr{i}" for i in range(1, 18)} | {"chrX", "chrY"}
VAL_CHROMS = {"chr18", "chr19"}
TEST_CHROMS = {"chr20", "chr21", "chr22"}
VALID_BASES = {"A", "C", "G", "T"}
META_COLUMNS = [
    "eid",
    "chrom",
    "start",
    "end",
    "state_int",
    "state_name",
    "family",
    "subcluster",
    "gc_content",
    "cpg_ratio",
    "entropy",
]

STATE_METADATA: Dict[str, Dict[str, object]] = {
    "1_TssA": {"state_int": 1, "family": "promoter", "subcluster": 1},
    "2_TssFlnk": {"state_int": 2, "family": "promoter", "subcluster": 1},
    "3_TssFlnkU": {"state_int": 3, "family": "promoter", "subcluster": 1},
    "4_TssFlnkD": {"state_int": 4, "family": "promoter", "subcluster": 1},
    "5_Tx": {"state_int": 5, "family": "transcribed", "subcluster": 3},
    "6_TxWk": {"state_int": 6, "family": "transcribed", "subcluster": 3},
    "7_EnhG1": {"state_int": 7, "family": "enhancer", "subcluster": 4},
    "8_EnhG2": {"state_int": 8, "family": "enhancer", "subcluster": 4},
    "9_EnhA1": {"state_int": 9, "family": "enhancer", "subcluster": 5},
    "10_EnhA2": {"state_int": 10, "family": "enhancer", "subcluster": 5},
    "11_EnhWk": {"state_int": 11, "family": "enhancer", "subcluster": 5},
    "12_ZNF/Rpts": {"state_int": 12, "family": "heterochromatin", "subcluster": 7},
    "13_Het": {"state_int": 13, "family": "heterochromatin", "subcluster": 7},
    "14_TssBiv": {"state_int": 14, "family": "promoter", "subcluster": 2},
    "15_EnhBiv": {"state_int": 15, "family": "enhancer", "subcluster": 5},
    "16_ReprPC": {"state_int": 16, "family": "polycomb", "subcluster": 6},
    "17_ReprPCWk": {"state_int": 17, "family": "polycomb", "subcluster": 6},
    "18_Quies": {"state_int": 18, "family": "quiescent", "subcluster": 7},
}
VALID_STATE_NAMES = set(STATE_METADATA)


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    phase_root: Path
    metadata_tsv_url: str
    bed_base_url: str
    chain_url: str
    hg38_fa_gz_url: str
    hg19_fa_gz_url: str
    liftover_binary_url: str
    liftover_min_match: float
    download_workers: int
    per_state_cap: int
    entropy_threshold: float
    minority_states: tuple[int, ...]
    reverse_complement_minority_states: bool
    cleanup_plain_beds_after_compression: bool
    keep_hg19_reference_gz: bool
    force_rebuild_merged: bool
    random_seed: int

    @classmethod
    def from_json(cls, config_path: Path, repo_root: Path) -> "PipelineConfig":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        required_keys = {
            "phase_root",
            "metadata_tsv_url",
            "bed_base_url",
            "chain_url",
            "hg38_fa_gz_url",
            "hg19_fa_gz_url",
            "liftover_binary_url",
            "liftover_min_match",
            "download_workers",
            "per_state_cap",
            "entropy_threshold",
            "minority_states",
            "reverse_complement_minority_states",
            "cleanup_plain_beds_after_compression",
            "keep_hg19_reference_gz",
            "force_rebuild_merged",
            "random_seed",
        }
        missing = sorted(required_keys - set(raw))
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")
        return cls(
            repo_root=repo_root,
            phase_root=repo_root / raw["phase_root"],
            metadata_tsv_url=raw["metadata_tsv_url"],
            bed_base_url=raw["bed_base_url"].rstrip("/"),
            chain_url=raw["chain_url"],
            hg38_fa_gz_url=raw["hg38_fa_gz_url"],
            hg19_fa_gz_url=raw["hg19_fa_gz_url"],
            liftover_binary_url=raw["liftover_binary_url"],
            liftover_min_match=float(raw["liftover_min_match"]),
            download_workers=int(raw["download_workers"]),
            per_state_cap=int(raw["per_state_cap"]),
            entropy_threshold=float(raw["entropy_threshold"]),
            minority_states=tuple(int(state) for state in raw["minority_states"]),
            reverse_complement_minority_states=bool(raw["reverse_complement_minority_states"]),
            cleanup_plain_beds_after_compression=bool(raw["cleanup_plain_beds_after_compression"]),
            keep_hg19_reference_gz=bool(raw["keep_hg19_reference_gz"]),
            force_rebuild_merged=bool(raw["force_rebuild_merged"]),
            random_seed=int(raw["random_seed"]),
        )


@dataclass
class ProcessedRecord:
    sequence: str
    metadata: Dict[str, object]


class RoadmapAggregationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.paths = self._build_paths(config.phase_root)
        configure_logging(log_dir=str(self.paths["logs"]), console_level=20)
        self.logger = get_logger("phase0_aggregate", log_file="phase0_aggregate.log")
        self.random = random.Random(config.random_seed)
        self._ensure_directories()

    def _build_paths(self, phase_root: Path) -> Dict[str, Path]:
        return {
            "phase_root": phase_root,
            "raw_beds": phase_root / "raw" / "beds",
            "raw_metadata": phase_root / "raw" / "metadata",
            "reference": phase_root / "reference",
            "normalized": phase_root / "intermediate" / "normalized",
            "lifted": phase_root / "intermediate" / "lifted",
            "unmapped": phase_root / "intermediate" / "unmapped",
            "per_epigenome": phase_root / "processed" / "per_epigenome",
            "merged": phase_root / "processed" / "merged",
            "logs": phase_root / "logs",
            "qc": phase_root / "logs" / "qc",
            "tools": phase_root / "tools",
        }

    def _ensure_directories(self) -> None:
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self.logger.info("Starting phase0 Roadmap aggregation pipeline")
        eids = self.fetch_remote_eids()
        metadata_df = self.download_and_prepare_metadata(eids)
        self.download_references()
        self.download_beds(eids)
        fasta = self.load_reference_fasta()
        chrom_sizes = self.load_chrom_sizes()
        for eid in eids:
            self.process_epigenome(eid=eid, metadata_row=metadata_df.loc[eid], fasta=fasta, chrom_sizes=chrom_sizes)
        fasta.close()
        self.write_qc_summary(metadata_df)
        self.build_merged_dataset(eids)
        self.write_pipeline_manifest(eids)
        self.logger.info("Completed phase0 Roadmap aggregation pipeline")

    def fetch_remote_eids(self) -> List[str]:
        url = f"{self.config.bed_base_url}/"
        self.logger.info("Fetching Roadmap BED inventory from %s", url)
        with urllib.request.urlopen(url, timeout=60) as response:
            html = response.read().decode("utf-8", errors="ignore")
        filenames = sorted(set(__import__("re").findall(r'href="(E\d{3}_18_core_K27ac_mnemonics\.bed\.gz)"', html)))
        if not filenames:
            raise RuntimeError(f"No 18-state Roadmap BED files found at {url}")
        eids = [filename.split("_", 1)[0] for filename in filenames]
        self.logger.info("Discovered %d Roadmap epigenomes in the 18-state release", len(eids))
        return eids

    def download_and_prepare_metadata(self, eids: Sequence[str]) -> pd.DataFrame:
        metadata_path = self.paths["raw_metadata"] / "roadmap_consolidated_metadata.tsv"
        self.download_file(self.config.metadata_tsv_url, metadata_path)
        metadata = pd.read_csv(metadata_path, sep="\t")
        metadata.columns = [str(column).strip() for column in metadata.columns]
        eid_column = "Epigenome ID (EID)"
        if eid_column not in metadata.columns:
            raise KeyError(f"Metadata table does not contain {eid_column!r}")
        metadata = metadata[metadata[eid_column].isin(eids)].copy()
        metadata["EID"] = metadata[eid_column]
        metadata["MARKS"] = "core_marks_plus_k27ac"
        metadata["Epigenome name"] = metadata["Standardized Epigenome name"].fillna(metadata["Epigenome name (from EDACC Release 9 directory)"])
        curated = metadata[
            [
                "EID",
                "Epigenome name",
                "GROUP",
                "TYPE",
                "ANATOMY",
                "MARKS",
                "Has K27ac",
                "CLASS",
            ]
        ].copy()
        curated.rename(columns={"ANATOMY": "Anatomy"}, inplace=True)
        curated.sort_values("EID", inplace=True)
        curated.to_csv(self.paths["raw_metadata"] / "roadmap_curated_metadata.tsv", sep="\t", index=False)
        curated.set_index("EID", inplace=True)
        missing = sorted(set(eids) - set(curated.index))
        if missing:
            raise RuntimeError(f"Metadata rows missing for epigenomes: {missing[:10]}")
        return curated

    def download_references(self) -> None:
        downloads = [
            (self.config.chain_url, self.paths["reference"] / "hg19ToHg38.over.chain.gz"),
            (self.config.hg38_fa_gz_url, self.paths["reference"] / "hg38.fa.gz"),
            (self.config.hg19_fa_gz_url, self.paths["reference"] / "hg19.fa.gz"),
            (self.config.liftover_binary_url, self.paths["tools"] / "liftOver"),
        ]
        for url, destination in downloads:
            self.download_file(url, destination)
        liftover_path = self.paths["tools"] / "liftOver"
        liftover_path.chmod(0o755)

        hg38_fa_path = self.paths["reference"] / "hg38.fa"
        if not hg38_fa_path.exists():
            self.logger.info("Decompressing hg38 reference FASTA")
            self.decompress_gzip_file(self.paths["reference"] / "hg38.fa.gz", hg38_fa_path)

        if not self.config.keep_hg19_reference_gz:
            hg19_path = self.paths["reference"] / "hg19.fa.gz"
            if hg19_path.exists():
                hg19_path.unlink()

    def download_beds(self, eids: Sequence[str]) -> None:
        tasks = []
        with ThreadPoolExecutor(max_workers=self.config.download_workers) as executor:
            for eid in eids:
                destination = self.paths["raw_beds"] / f"{eid}_18_core_K27ac_mnemonics.bed.gz"
                url = f"{self.config.bed_base_url}/{destination.name}"
                tasks.append(executor.submit(self.download_file, url, destination))
            for future in as_completed(tasks):
                future.result()

    def load_reference_fasta(self) -> Fasta:
        hg38_fa_path = self.paths["reference"] / "hg38.fa"
        self.logger.info("Loading hg38 reference FASTA from %s", hg38_fa_path)
        fasta = Fasta(
            str(hg38_fa_path),
            as_raw=True,
            build_index=True,
            sequence_always_upper=True,
        )
        return fasta

    def load_chrom_sizes(self) -> Dict[str, int]:
        fai_path = self.paths["reference"] / "hg38.fa.fai"
        if not fai_path.exists():
            raise FileNotFoundError(f"Missing hg38 FASTA index: {fai_path}")
        chrom_sizes: Dict[str, int] = {}
        with fai_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                chrom, size, *_ = line.rstrip().split("\t")
                chrom_sizes[chrom] = int(size)
        return chrom_sizes

    def process_epigenome(
        self,
        eid: str,
        metadata_row: pd.Series,
        fasta: Fasta,
        chrom_sizes: Dict[str, int],
    ) -> None:
        outputs = self.epigenome_output_paths(eid)
        if all(path.exists() for path in outputs.values()):
            self.logger.info("Skipping %s because processed outputs already exist", eid)
            return

        raw_bed_path = self.paths["raw_beds"] / f"{eid}_18_core_K27ac_mnemonics.bed.gz"
        normalized_plain = self.paths["normalized"] / f"{eid}.normalized.bed"
        lifted_plain = self.paths["lifted"] / f"{eid}.lifted.bed"
        unmapped_plain = self.paths["unmapped"] / f"{eid}.unmapped.bed"

        self.logger.info("Processing %s (%s)", eid, metadata_row["Epigenome name"])
        normalization_metrics = self.normalize_bed(raw_bed_path, normalized_plain)
        self.run_liftover(normalized_plain, lifted_plain, unmapped_plain)
        extraction_metrics = self.extract_sequences(
            eid=eid,
            lifted_plain=lifted_plain,
            fasta=fasta,
            chrom_sizes=chrom_sizes,
            metadata_row=metadata_row,
        )
        self.compress_if_requested(normalized_plain)
        self.compress_if_requested(lifted_plain)
        self.compress_if_requested(unmapped_plain)

        summary = {
            "eid": eid,
            "epigenome_name": metadata_row["Epigenome name"],
            "group": metadata_row["GROUP"],
            "type": metadata_row["TYPE"],
            "anatomy": metadata_row["Anatomy"],
            "normalization": normalization_metrics,
            "extraction": extraction_metrics,
            "raw_bed_size_mb": round(raw_bed_path.stat().st_size / (1024 * 1024), 2),
        }
        with outputs["summary_json"].open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        log_metrics(
            self.logger,
            {
                "eid": eid,
                "normalized_rows": normalization_metrics["expanded_bins_written"],
                "surviving_rows": extraction_metrics["surviving_rows"],
            },
            message="Per-epigenome completion",
        )

    def epigenome_output_paths(self, eid: str) -> Dict[str, Path]:
        return {
            "sequences_csv": self.paths["per_epigenome"] / f"{eid}_sequences.csv",
            "labels_csv": self.paths["per_epigenome"] / f"{eid}_labels.csv",
            "meta_csv_gz": self.paths["per_epigenome"] / f"{eid}_meta.csv.gz",
            "summary_json": self.paths["qc"] / f"{eid}_summary.json",
        }

    def normalize_bed(self, raw_bed_path: Path, normalized_plain: Path) -> Dict[str, object]:
        tmp_path = normalized_plain.with_suffix(".tmp")
        metrics = {
            "raw_segments": 0,
            "expanded_bins_written": 0,
            "invalid_state_segments": 0,
            "noncanonical_segments": 0,
            "malformed_segments": 0,
            "non_multiple_200_segments": 0,
            "sort_violations": 0,
            "chromosomes_seen": [],
        }
        chroms_seen = set()
        prev_chrom = ""
        prev_start = -1
        with gzip.open(raw_bed_path, "rt", encoding="utf-8") as input_handle, tmp_path.open("w", encoding="utf-8") as output_handle:
            for line in input_handle:
                metrics["raw_segments"] += 1
                parts = line.rstrip().split("\t")
                if len(parts) < 4:
                    metrics["malformed_segments"] += 1
                    continue
                chrom, start_text, end_text, state_name = parts[:4]
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                if state_name not in VALID_STATE_NAMES:
                    metrics["invalid_state_segments"] += 1
                    continue
                if chrom not in CANONICAL_CHROM_SET:
                    metrics["noncanonical_segments"] += 1
                    continue
                start = int(start_text)
                end = int(end_text)
                segment_length = end - start
                if segment_length <= 0 or segment_length % 200 != 0:
                    metrics["non_multiple_200_segments"] += 1
                    continue
                chroms_seen.add(chrom)
                if prev_chrom and (chrom < prev_chrom or (chrom == prev_chrom and start < prev_start)):
                    metrics["sort_violations"] += 1
                prev_chrom = chrom
                prev_start = start
                for bin_start in range(start, end, 200):
                    output_handle.write(f"{chrom}\t{bin_start}\t{bin_start + 200}\t{state_name}\n")
                    metrics["expanded_bins_written"] += 1
        metrics["chromosomes_seen"] = sorted(chroms_seen)
        if metrics["sort_violations"] > 0:
            self.logger.info("Sorting normalized BED because %d order violations were detected", metrics["sort_violations"])
            subprocess.run(
                [
                    "sort",
                    "-k1,1",
                    "-k2,2n",
                    "-s",
                    str(tmp_path),
                    "-o",
                    str(normalized_plain),
                ],
                check=True,
            )
            tmp_path.unlink()
        else:
            tmp_path.replace(normalized_plain)
        return metrics

    def run_liftover(self, normalized_plain: Path, lifted_plain: Path, unmapped_plain: Path) -> None:
        command = [
            str(self.paths["tools"] / "liftOver"),
            f"-minMatch={self.config.liftover_min_match}",
            str(normalized_plain),
            str(self.paths["reference"] / "hg19ToHg38.over.chain.gz"),
            str(lifted_plain),
            str(unmapped_plain),
        ]
        self.logger.info("Running liftOver on %s", normalized_plain.name)
        subprocess.run(command, check=True)

    def extract_sequences(
        self,
        eid: str,
        lifted_plain: Path,
        fasta: Fasta,
        chrom_sizes: Dict[str, int],
        metadata_row: pd.Series,
    ) -> Dict[str, object]:
        outputs = self.epigenome_output_paths(eid)
        temp_sequences = outputs["sequences_csv"].with_suffix(".csv.tmp")
        temp_labels = outputs["labels_csv"].with_suffix(".csv.tmp")
        temp_meta = outputs["meta_csv_gz"].with_name(outputs["meta_csv_gz"].name + ".tmp")
        state_counts = {str(meta["state_int"]): 0 for meta in STATE_METADATA.values()}
        split_counts = {"train": 0, "val": 0, "test": 0}
        gc_by_state: Dict[int, List[float]] = {int(meta["state_int"]): [] for meta in STATE_METADATA.values()}
        metrics = {
            "lifted_rows_seen": 0,
            "post_liftover_noncanonical": 0,
            "post_liftover_bad_length": 0,
            "post_liftover_out_of_bounds": 0,
            "overlap_rows_dropped": 0,
            "n_filtered": 0,
            "invalid_alphabet_filtered": 0,
            "entropy_filtered": 0,
            "surviving_rows": 0,
            "state_counts": state_counts,
            "split_counts": split_counts,
            "gc_mean_by_state": {},
        }

        with temp_sequences.open("w", encoding="utf-8") as sequence_handle, temp_labels.open("w", encoding="utf-8") as label_handle, gzip.open(temp_meta, "wt", encoding="utf-8", newline="") as meta_handle:
            meta_writer = csv.DictWriter(meta_handle, fieldnames=META_COLUMNS)
            meta_writer.writeheader()

            pending_row: tuple[str, int, int, str] | None = None
            with lifted_plain.open("r", encoding="utf-8") as handle:
                for line in handle:
                    metrics["lifted_rows_seen"] += 1
                    parts = line.rstrip().split("\t")
                    if len(parts) < 4:
                        continue
                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    state_name = parts[3]
                    if chrom not in CANONICAL_CHROM_SET:
                        metrics["post_liftover_noncanonical"] += 1
                        continue
                    if end - start != 200:
                        metrics["post_liftover_bad_length"] += 1
                        continue
                    if chrom not in chrom_sizes or end > chrom_sizes[chrom]:
                        metrics["post_liftover_out_of_bounds"] += 1
                        continue
                    current = (chrom, start, end, state_name)
                    if pending_row is not None and chrom == pending_row[0] and start < pending_row[2]:
                        metrics["overlap_rows_dropped"] += 2
                        pending_row = None
                        continue
                    if pending_row is not None:
                        self._emit_record(
                            eid=eid,
                            row=pending_row,
                            fasta=fasta,
                            sequence_handle=sequence_handle,
                            label_handle=label_handle,
                            meta_writer=meta_writer,
                            metrics=metrics,
                            gc_by_state=gc_by_state,
                        )
                    pending_row = current
            if pending_row is not None:
                self._emit_record(
                    eid=eid,
                    row=pending_row,
                    fasta=fasta,
                    sequence_handle=sequence_handle,
                    label_handle=label_handle,
                    meta_writer=meta_writer,
                    metrics=metrics,
                    gc_by_state=gc_by_state,
                )

        for state_int, gc_values in gc_by_state.items():
            if gc_values:
                metrics["gc_mean_by_state"][str(state_int)] = round(sum(gc_values) / len(gc_values), 6)
        temp_sequences.replace(outputs["sequences_csv"])
        temp_labels.replace(outputs["labels_csv"])
        temp_meta.replace(outputs["meta_csv_gz"])
        return metrics

    def _emit_record(
        self,
        eid: str,
        row: tuple[str, int, int, str],
        fasta: Fasta,
        sequence_handle,
        label_handle,
        meta_writer: csv.DictWriter,
        metrics: Dict[str, object],
        gc_by_state: Dict[int, List[float]],
    ) -> None:
        chrom, start, end, state_name = row
        sequence = fasta[chrom][start:end].upper()
        if len(sequence) != 200:
            metrics["post_liftover_bad_length"] += 1
            return
        if "N" in sequence:
            metrics["n_filtered"] += 1
            return
        if any(base not in VALID_BASES for base in sequence):
            metrics["invalid_alphabet_filtered"] += 1
            return
        entropy = shannon_entropy(sequence)
        if entropy <= self.config.entropy_threshold:
            metrics["entropy_filtered"] += 1
            return

        state_info = STATE_METADATA[state_name]
        state_int = int(state_info["state_int"])
        gc_content = gc_fraction(sequence)
        cpg_ratio = observed_expected_cpg(sequence)
        split_name = split_for_chromosome(chrom)

        sequence_handle.write(f"{sequence}\n")
        label_handle.write(f"{state_int}\n")
        meta_writer.writerow(
            {
                "eid": eid,
                "chrom": chrom,
                "start": start,
                "end": end,
                "state_int": state_int,
                "state_name": state_name,
                "family": state_info["family"],
                "subcluster": state_info["subcluster"],
                "gc_content": f"{gc_content:.6f}",
                "cpg_ratio": f"{cpg_ratio:.6f}",
                "entropy": f"{entropy:.6f}",
            }
        )
        metrics["surviving_rows"] += 1
        metrics["state_counts"][str(state_int)] += 1
        metrics["split_counts"][split_name] += 1
        gc_by_state[state_int].append(gc_content)

    def build_merged_dataset(self, eids: Sequence[str]) -> None:
        merged_dir = self.paths["merged"]
        if self.config.force_rebuild_merged:
            for path in merged_dir.glob("*"):
                if path.is_file():
                    path.unlink()
        merged_paths = {
            "train_sequences": merged_dir / "train_sequences.csv",
            "train_labels": merged_dir / "train_labels.csv",
            "train_meta": merged_dir / "train_meta.csv",
            "val_sequences": merged_dir / "val_sequences.csv",
            "val_labels": merged_dir / "val_labels.csv",
            "val_meta": merged_dir / "val_meta.csv",
            "test_sequences": merged_dir / "test_sequences.csv",
            "test_labels": merged_dir / "test_labels.csv",
            "test_meta": merged_dir / "test_meta.csv",
        }
        state_cap_per_epigenome = max(1, self.config.per_state_cap // len(eids))
        self.logger.info(
            "Building merged dataset with per-state per-epigenome training cap of %d",
            state_cap_per_epigenome,
        )
        with (
            merged_paths["train_sequences"].open("w", encoding="utf-8") as train_seq,
            merged_paths["train_labels"].open("w", encoding="utf-8") as train_lab,
            merged_paths["train_meta"].open("w", encoding="utf-8", newline="") as train_meta_handle,
            merged_paths["val_sequences"].open("w", encoding="utf-8") as val_seq,
            merged_paths["val_labels"].open("w", encoding="utf-8") as val_lab,
            merged_paths["val_meta"].open("w", encoding="utf-8", newline="") as val_meta_handle,
            merged_paths["test_sequences"].open("w", encoding="utf-8") as test_seq,
            merged_paths["test_labels"].open("w", encoding="utf-8") as test_lab,
            merged_paths["test_meta"].open("w", encoding="utf-8", newline="") as test_meta_handle,
        ):
            train_meta_writer = csv.DictWriter(train_meta_handle, fieldnames=META_COLUMNS)
            val_meta_writer = csv.DictWriter(val_meta_handle, fieldnames=META_COLUMNS)
            test_meta_writer = csv.DictWriter(test_meta_handle, fieldnames=META_COLUMNS)
            train_meta_writer.writeheader()
            val_meta_writer.writeheader()
            test_meta_writer.writeheader()

            merged_counts = {
                "train": {state: 0 for state in range(1, 19)},
                "val": {state: 0 for state in range(1, 19)},
                "test": {state: 0 for state in range(1, 19)},
            }

            for eid in eids:
                outputs = self.epigenome_output_paths(eid)
                train_reservoirs: Dict[int, List[ProcessedRecord]] = {state: [] for state in range(1, 19)}
                train_seen: Dict[int, int] = {state: 0 for state in range(1, 19)}
                with (
                    outputs["sequences_csv"].open("r", encoding="utf-8") as seq_handle,
                    outputs["labels_csv"].open("r", encoding="utf-8") as label_handle,
                    gzip.open(outputs["meta_csv_gz"], "rt", encoding="utf-8", newline="") as meta_handle,
                ):
                    meta_reader = csv.DictReader(meta_handle)
                    for sequence_line, label_line, meta_row in zip(seq_handle, label_handle, meta_reader):
                        sequence = sequence_line.rstrip("\n")
                        state_int = int(label_line.strip())
                        split_name = split_for_chromosome(meta_row["chrom"])
                        record = ProcessedRecord(sequence=sequence, metadata=meta_row)
                        if split_name == "train":
                            train_seen[state_int] += 1
                            reservoir = train_reservoirs[state_int]
                            if len(reservoir) < state_cap_per_epigenome:
                                reservoir.append(record)
                            else:
                                replace_idx = self.random.randint(0, train_seen[state_int] - 1)
                                if replace_idx < state_cap_per_epigenome:
                                    reservoir[replace_idx] = record
                            continue
                        writer_seq, writer_lab, writer_meta = (
                            (val_seq, val_lab, val_meta_writer)
                            if split_name == "val"
                            else (test_seq, test_lab, test_meta_writer)
                        )
                        writer_seq.write(f"{sequence}\n")
                        writer_lab.write(f"{state_int}\n")
                        writer_meta.writerow(meta_row)
                        merged_counts[split_name][state_int] += 1

                for state_int, records in train_reservoirs.items():
                    for record in records:
                        train_seq.write(f"{record.sequence}\n")
                        train_lab.write(f"{state_int}\n")
                        train_meta_writer.writerow(record.metadata)
                        merged_counts["train"][state_int] += 1
                        if self.config.reverse_complement_minority_states and state_int in self.config.minority_states:
                            train_seq.write(f"{reverse_complement(record.sequence)}\n")
                            train_lab.write(f"{state_int}\n")
                            train_meta_writer.writerow(record.metadata)
                            merged_counts["train"][state_int] += 1

        manifest = {
            "per_state_cap": self.config.per_state_cap,
            "effective_train_cap_per_epigenome_state": state_cap_per_epigenome,
            "minority_states": list(self.config.minority_states),
            "reverse_complement_minority_states": self.config.reverse_complement_minority_states,
            "split_state_counts": {
                split: {str(state): count for state, count in counts.items()}
                for split, counts in merged_counts.items()
            },
        }
        with (self.paths["qc"] / "merged_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.competition_format_check(merged_paths)

    def competition_format_check(self, merged_paths: Dict[str, Path]) -> None:
        rng = random.Random(self.config.random_seed)
        compatibility = {}
        for split_name in ("train", "val", "test"):
            sequences_path = merged_paths[f"{split_name}_sequences"]
            labels_path = merged_paths[f"{split_name}_labels"]
            sample_sequences = reservoir_sample_lines(sequences_path, 100, rng)
            sample_labels = reservoir_sample_lines(labels_path, 100, rng)
            compatibility[split_name] = {
                "sampled_sequences": len(sample_sequences),
                "sampled_labels": len(sample_labels),
                "all_sequences_length_200": all(len(seq) == 200 for seq in sample_sequences),
                "all_sequences_acgt_only": all(set(seq).issubset(VALID_BASES) for seq in sample_sequences),
                "labels_in_expected_range": all(1 <= int(label) <= 18 for label in sample_labels),
            }
        with (self.paths["qc"] / "competition_format_check.json").open("w", encoding="utf-8") as handle:
            json.dump(compatibility, handle, indent=2)

    def write_qc_summary(self, metadata_df: pd.DataFrame) -> None:
        summaries = []
        for summary_path in sorted(self.paths["qc"].glob("E*_summary.json")):
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            state_counts = {int(k): int(v) for k, v in summary["extraction"]["state_counts"].items()}
            total = max(1, summary["extraction"]["surviving_rows"])
            summary_row = {
                "eid": summary["eid"],
                "epigenome_name": summary["epigenome_name"],
                "group": summary["group"],
                "type": summary["type"],
                "anatomy": summary["anatomy"],
                "raw_segments": summary["normalization"]["raw_segments"],
                "expanded_bins": summary["normalization"]["expanded_bins_written"],
                "surviving_rows": summary["extraction"]["surviving_rows"],
                "quies_fraction": round(state_counts.get(18, 0) / total, 6),
                "tssa_fraction": round(state_counts.get(1, 0) / total, 6),
                "normalization_drop_fraction": round(
                    (
                        summary["normalization"]["invalid_state_segments"]
                        + summary["normalization"]["noncanonical_segments"]
                        + summary["normalization"]["malformed_segments"]
                        + summary["normalization"]["non_multiple_200_segments"]
                    )
                    / max(1, summary["normalization"]["raw_segments"]),
                    6,
                ),
            }
            summaries.append(summary_row)
        pd.DataFrame(summaries).sort_values("eid").to_csv(self.paths["qc"] / "per_epigenome_qc.csv", index=False)
        metadata_df.reset_index().to_csv(self.paths["qc"] / "metadata_snapshot.csv", index=False)

    def write_pipeline_manifest(self, eids: Sequence[str]) -> None:
        manifest = {
            "release_type": "Roadmap 18-state core_K27ac",
            "epigenome_count": len(eids),
            "phase_root": str(self.config.phase_root),
            "bed_base_url": self.config.bed_base_url,
            "metadata_tsv_url": self.config.metadata_tsv_url,
            "chain_url": self.config.chain_url,
            "hg38_fa_gz_url": self.config.hg38_fa_gz_url,
            "liftover_binary_url": self.config.liftover_binary_url,
        }
        with (self.paths["phase_root"] / "pipeline_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def download_file(self, url: str, destination: Path) -> Path:
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(destination.suffix + ".part")
        self.logger.info("Downloading %s -> %s", url, destination)
        with urllib.request.urlopen(url, timeout=120) as response, tmp_destination.open("wb") as output_handle:
            shutil.copyfileobj(response, output_handle, length=1024 * 1024)
        tmp_destination.replace(destination)
        return destination

    def compress_if_requested(self, plain_path: Path) -> None:
        if not plain_path.exists():
            return
        gz_path = plain_path.with_suffix(plain_path.suffix + ".gz")
        if not gz_path.exists():
            with plain_path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        if self.config.cleanup_plain_beds_after_compression:
            plain_path.unlink()

    def decompress_gzip_file(self, source_gz: Path, destination_plain: Path) -> None:
        tmp_destination = destination_plain.with_suffix(destination_plain.suffix + ".tmp")
        with gzip.open(source_gz, "rb") as source_handle, tmp_destination.open("wb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        tmp_destination.replace(destination_plain)


def reservoir_sample_lines(path: Path, sample_size: int, rng: random.Random) -> List[str]:
    reservoir: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            value = line.rstrip("\n")
            if len(reservoir) < sample_size:
                reservoir.append(value)
                continue
            replace_idx = rng.randint(0, idx - 1)
            if replace_idx < sample_size:
                reservoir[replace_idx] = value
    return reservoir


def split_for_chromosome(chrom: str) -> str:
    if chrom in TRAIN_CHROMS:
        return "train"
    if chrom in VAL_CHROMS:
        return "val"
    if chrom in TEST_CHROMS:
        return "test"
    raise ValueError(f"Chromosome {chrom} does not belong to a configured split")


def gc_fraction(sequence: str) -> float:
    gc = sequence.count("G") + sequence.count("C")
    return gc / len(sequence)


def observed_expected_cpg(sequence: str) -> float:
    c_count = sequence.count("C")
    g_count = sequence.count("G")
    expected = (c_count * g_count) / len(sequence)
    if expected == 0:
        return 0.0
    observed = sequence.count("CG")
    return observed / expected


def shannon_entropy(sequence: str) -> float:
    length = len(sequence)
    entropy = 0.0
    for base in VALID_BASES:
        probability = sequence.count(base) / length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def reverse_complement(sequence: str) -> str:
    translation = str.maketrans("ACGT", "TGCA")
    return sequence.translate(translation)[::-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Roadmap 18-state aggregation pipeline.")
    parser.add_argument("--config", required=True, help="Path to the explicit JSON configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = PipelineConfig.from_json(config_path=config_path, repo_root=repo_root)
    pipeline = RoadmapAggregationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
