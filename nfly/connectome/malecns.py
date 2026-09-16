"""Load the Janelia MaleCNS v1.0 connectome (https://male-cns.janelia.org/download/).

Files (public, Google Storage, Apache Arrow feather):
    body-annotations       one row per body (segment); neurons are the bodies with a superclass
    body-neurotransmitters per-body transmitter prediction (consensus_nt)
    connectome-weights     body -> body synapse counts (minconf >= 0.5), 152M rows incl. fragments

Mapping onto the Connectome tensors used by ConnectomeRNN:
    bodyId                           -> node index 0..N-1
    (body_pre, body_post, weight)    -> W[post, pre], magnitude = weight / total input of post
    consensus_nt of presynaptic body -> sign (Dale's law): ACh/DA/5-HT/OA +, GABA/Glu/His -
    superclass                       -> flow: afferent / intrinsic / efferent
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .base import Connectome, build_connectome

log = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclasses.dataclass(frozen=True)
class DataFile:
    local: str      # short name we look for in data_dir
    remote: str     # file name on Google Storage

    def path(self, data_dir: Path) -> Path:
        for name in (self.local, self.remote):
            if (data_dir / name).exists():
                return data_dir / name
        raise FileNotFoundError(f"{self.local} not found in {data_dir}; download {BASE_URL}/{self.remote}")


ANNOTATIONS = DataFile("body-annotations.feather", "body-annotations-male-cns-v1.0-minconf-0.5.feather")
TRANSMITTERS = DataFile("body-neurotransmitters.feather", "body-neurotransmitters-male-cns-v1.0.feather")
WEIGHTS = DataFile("connectome-weights.feather", "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

# Drosophila sign convention.  Histamine (photoreceptors) acts through HisCl chloride channels
# and glutamate mostly through GluCl, so both count as inhibitory.
DEFAULT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
                "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
NT_SHORT = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT", "dopamine": "DA",
            "serotonin": "SER", "octopamine": "OCT", "histamine": "HIS", "unclear": "UNKNOWN"}

# superclass -> flow.  Sensory neurons are the network's inputs; motor / efferent / endocrine
# are its outputs; everything else (incl. descending and ascending neurons, which stay inside
# the CNS) is hidden.
AFFERENT = {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"}
EFFERENT = {"vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_endocrine", "cb_endocrine",
            "efferent_ascending", "efferent_descending"}

ANNOTATION_COLUMNS = {"bodyId": "root_id", "superclass": "super_class", "class": "class", "subclass": "sub_class",
                      "type": "cell_type", "flywireType": "flywire_type", "hemibrainType": "hemibrain_type",
                      "somaSide": "side", "instance": "instance", "statusLabel": "status_label",
                      "dimorphism": "dimorphism", "fruDsx": "fru_dsx", "entryNerve": "entry_nerve",
                      "exitNerve": "exit_nerve"}
HEX_COLUMNS = {"assignedOlHex1": "hex1", "assignedOlHex2": "hex2"}
NO_HEX = np.float32(-1)
POSITION_COLUMN = "somaLocation"     # [x, y, z] in 8 nm voxels of the MaleCNS EM space
VOXEL_NM = 8.0
CACHE_VERSION = 2                    # bump when the neuron table gains columns


def flow_of(superclass: pd.Series) -> pd.Series:
    out = pd.Series("intrinsic", index=superclass.index)
    out[superclass.isin(AFFERENT)] = "afferent"
    out[superclass.isin(EFFERENT)] = "efferent"
    return out


def _read_annotations(data_dir: Path) -> pd.DataFrame:
    """Annotated neurons (bodies with a superclass) with unified column names and hex coordinates."""
    ann = feather.read_feather(ANNOTATIONS.path(data_dir))
    ann = ann[ann["superclass"].notna()]
    present = {k: v for k, v in ANNOTATION_COLUMNS.items() if k in ann.columns}
    df = ann[list(present)].rename(columns=present)
    df["flow"] = flow_of(df["super_class"])
    for src, dst in HEX_COLUMNS.items():
        df[dst] = ann[src].fillna(NO_HEX).astype(np.float32).to_numpy() if src in ann.columns else NO_HEX
    df[["x", "y", "z"]] = _soma_xyz_um(ann)
    return df


def _soma_xyz_um(ann: pd.DataFrame) -> np.ndarray:
    """Soma position per neuron in micrometres, NaN where the release has none (sensory neurons
    whose somata lie outside the imaged volume)."""
    xyz = np.full((len(ann), 3), np.nan, dtype=np.float32)
    if POSITION_COLUMN in ann.columns:
        loc = ann[POSITION_COLUMN]
        ok = loc.map(lambda v: v is not None and len(v) == 3).to_numpy()
        xyz[ok] = np.stack(loc[ok].to_numpy()).astype(np.float32) * (VOXEL_NM / 1000.0)
    return xyz


def _read_transmitters(data_dir: Path) -> pd.DataFrame:
    nt = feather.read_feather(TRANSMITTERS.path(data_dir), columns=["body", "consensus_nt", "predicted_nt_confidence"])
    return nt.rename(columns={"body": "root_id", "consensus_nt": "nt_full", "predicted_nt_confidence": "nt_type_score"})


def _blank_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    return df


def read_neuron_table(data_dir: Path) -> pd.DataFrame:
    """One row per neuron: root_id, nt_type, flow, super_class, class, cell_type, side, hex1, hex2, ..."""
    df = _read_annotations(data_dir).merge(_read_transmitters(data_dir), on="root_id", how="left")
    df["nt_full"] = df["nt_full"].fillna("unclear")
    df["nt_type"] = df["nt_full"].map(NT_SHORT).fillna("UNKNOWN")
    df = _blank_missing_strings(df)
    df["root_id"] = df["root_id"].astype(np.int64)
    return df.sort_values("root_id").reset_index(drop=True)


def _pick(names, *candidates: str) -> str:
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"none of {candidates} in columns {list(names)}")


def read_weights(data_dir: Path, keep_ids: np.ndarray, min_syn: int = 1) -> pd.DataFrame:
    """(body_pre, body_post, syn_count) between `keep_ids` with at least `min_syn` synapses.

    The table has 152M rows including unannotated fragments, so it is filtered batch by batch
    in pyarrow before anything is materialised in pandas."""
    reader = pa.ipc.open_file(WEIGHTS.path(data_dir))
    names = reader.schema.names
    pre, post, wt = _pick(names, "body_pre", "pre"), _pick(names, "body_post", "post"), _pick(names, "weight", "syn_count")
    ids = pa.array(np.asarray(keep_ids, dtype=np.int64))
    parts = []
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i).select([pre, post, wt])
        mask = pc.and_(pc.is_in(b.column(pre), value_set=ids), pc.is_in(b.column(post), value_set=ids))
        if min_syn > 1:
            mask = pc.and_(mask, pc.greater_equal(b.column(wt), min_syn))
        b = b.filter(mask)
        if b.num_rows:
            parts.append(b)
    w = pa.Table.from_batches(parts).rename_columns(["body_pre", "body_post", "syn_count"]).to_pandas()
    if w.duplicated(["body_pre", "body_post"]).any():
        w = w.groupby(["body_pre", "body_post"], sort=False, as_index=False)["syn_count"].sum()
    return w.reset_index(drop=True)


def edge_signs(nt_full_pre: np.ndarray, sign_map: dict[str, float]) -> np.ndarray:
    """Dale's law: one sign per edge from the presynaptic neuron's transmitter; unknown -> +1."""
    known = np.isin(nt_full_pre, list(sign_map))
    if not known.all():
        log.warning("%s edges (%.1f%%) from neurons with unclear transmitter treated as excitatory",
                    f"{int((~known).sum()):,}", 100 * (~known).mean())
    lut = pd.Series(sign_map)
    return np.where(known, lut.reindex(nt_full_pre).fillna(1.0).to_numpy(), 1.0).astype(np.float32)


def load_malecns(data_dir: str | Path = "data", cache: bool = True, min_syn: int = 3,
                 sign_map: dict[str, float] | None = None) -> Connectome:
    """Parse the MaleCNS release (or a cached .pt) into a Connectome; `min_syn` drops weak pairs."""
    data_dir = Path(data_dir)
    cache_path = data_dir / "cache" / f"malecns_min{min_syn}_v{CACHE_VERSION}.pt"
    if cache and cache_path.exists():
        log.info("loading cached connectome from %s", cache_path)
        return Connectome.load(cache_path)

    neurons = read_neuron_table(data_dir)
    w = read_weights(data_dir, keep_ids=neurons["root_id"].to_numpy(), min_syn=min_syn)
    log.info("weights table: %s neuron->neuron pairs with >= %d synapses", f"{len(w):,}", min_syn)
    lut = pd.Series(np.arange(len(neurons)), index=neurons["root_id"].to_numpy())
    pre = lut.loc[w["body_pre"].to_numpy()].to_numpy()
    post = lut.loc[w["body_post"].to_numpy()].to_numpy()
    sign = edge_signs(neurons["nt_full"].to_numpy()[pre], sign_map or DEFAULT_SIGN)

    c = build_connectome(neurons, pre, post, w["syn_count"].to_numpy().astype(np.float32), sign)
    log.info(c.summary())
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        c.save(cache_path)
    return c
