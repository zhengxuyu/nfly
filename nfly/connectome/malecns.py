"""Load the Janelia MaleCNS v1.0 connectome (https://male-cns.janelia.org/download/).

Files (public, Google Storage, Apache Arrow feather):
    body-annotations-male-cns-v1.0-minconf-0.5.feather     one row per body (segment); neurons have a superclass
    body-neurotransmitters-male-cns-v1.0.feather           per-body transmitter prediction (consensus_nt)
    connectome-weights-male-cns-v1.0-minconf-0.5.feather   body -> body synapse counts (minconf >= 0.5)

Mapping onto the Connectome tensors used by ConnectomeRNN:
    bodyId                          -> node index 0..N-1        (only bodies with a superclass = neurons)
    (body_pre, body_post, weight)   -> W[post, pre]             magnitude = weight / total input of post
    consensus_nt of presynaptic body-> sign (Dale's law)        ACh/DA/5HT/OA +, GABA/Glu/His -
    superclass                      -> flow: afferent / intrinsic / efferent
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from .base import Connectome, build_connectome

log = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
LOCAL = {"annotations": "body-annotations.feather", "neurotransmitters": "body-neurotransmitters.feather",
         "weights": "connectome-weights.feather"}

# Drosophila sign convention.  Histamine (photoreceptors) acts through HisCl chloride channels
# and glutamate mostly through GluCl, so both count as inhibitory.
DEFAULT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
                "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
NT_SHORT = {"acetylcholine": "ACH", "gaba": "GABA", "glutamate": "GLUT", "dopamine": "DA",
            "serotonin": "SER", "octopamine": "OCT", "histamine": "HIS", "unclear": "UNKNOWN"}

# superclass -> flow.  Sensory neurons are the network's inputs; motor / efferent /
# endocrine are its outputs; everything else (incl. descending & ascending neurons, which
# stay inside the CNS) is hidden.
AFFERENT = {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"}
EFFERENT = {"vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_endocrine", "cb_endocrine",
            "efferent_ascending", "efferent_descending"}


def flow_of(superclass: pd.Series) -> pd.Series:
    out = pd.Series("intrinsic", index=superclass.index)
    out[superclass.isin(AFFERENT)] = "afferent"
    out[superclass.isin(EFFERENT)] = "efferent"
    return out


def _path(data_dir: Path, key: str) -> Path:
    for name in (LOCAL[key], FILES[key]):
        if (data_dir / name).exists():
            return data_dir / name
    raise FileNotFoundError(f"{LOCAL[key]} not found in {data_dir}; download {BASE_URL}/{FILES[key]}")


def read_neuron_table(data_dir: Path, neuron_filter: str = "superclass") -> pd.DataFrame:
    """One row per neuron with unified columns: root_id, nt_type, flow, super_class, class, cell_type, side ..."""
    ann = feather.read_feather(_path(data_dir, "annotations"))
    if neuron_filter == "superclass":
        ann = ann[ann["superclass"].notna()]
    ann = ann.rename(columns={"bodyId": "root_id", "superclass": "super_class", "type": "cell_type",
                              "subclass": "sub_class", "somaSide": "side", "hemibrainType": "hemibrain_type",
                              "flywireType": "flywire_type", "statusLabel": "status_label"})
    keep = ["root_id", "super_class", "class", "sub_class", "cell_type", "flywire_type", "hemibrain_type",
            "side", "instance", "status_label", "dimorphism", "fruDsx", "entryNerve", "exitNerve"]
    df = ann[[c for c in keep if c in ann.columns]].copy()
    df["flow"] = flow_of(df["super_class"])
    # optic-lobe column (ommatidium) hex coordinates, -1 where unassigned
    for k, col in (("hex1", "assignedOlHex1"), ("hex2", "assignedOlHex2")):
        df[k] = ann[col].fillna(-1).astype(np.float32).to_numpy() if col in ann.columns else np.float32(-1)

    nt = feather.read_feather(_path(data_dir, "neurotransmitters"),
                              columns=["body", "consensus_nt", "predicted_nt_confidence"])
    nt = nt.rename(columns={"body": "root_id", "consensus_nt": "nt_full", "predicted_nt_confidence": "nt_type_score"})
    df = df.merge(nt, on="root_id", how="left")
    df["nt_full"] = df["nt_full"].fillna("unclear")
    df["nt_type"] = df["nt_full"].map(NT_SHORT).fillna("UNKNOWN")
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    df["root_id"] = df["root_id"].astype(np.int64)
    return df.sort_values("root_id").reset_index(drop=True)


def _pick(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"none of {candidates} in columns {list(cols)}")


def read_weights(data_dir: Path, keep_ids: np.ndarray | None = None, min_syn: int = 1) -> pd.DataFrame:
    """Return body_pre, body_post, syn_count.

    The v1.0 table has 152M rows (body_pre, body_post, weight) including unannotated fragments,
    so filtering is done batch-wise in pyarrow before anything is materialised in pandas.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    reader = pa.ipc.open_file(_path(data_dir, "weights"))
    names = reader.schema.names
    pre = _pick(names, "body_pre", "bodyId_pre", "pre")
    post = _pick(names, "body_post", "bodyId_post", "post")
    wt = _pick(names, "weight", "syn_count", "count", "n")
    ids = pa.array(np.asarray(keep_ids, dtype=np.int64)) if keep_ids is not None else None
    parts = []
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i).select([pre, post, wt])
        mask = pc.greater_equal(b.column(wt), min_syn) if min_syn > 1 else None
        if ids is not None:
            m2 = pc.and_(pc.is_in(b.column(pre), value_set=ids), pc.is_in(b.column(post), value_set=ids))
            mask = m2 if mask is None else pc.and_(mask, m2)
        if mask is not None:
            b = b.filter(mask)
        if b.num_rows:
            parts.append(b)
    w = pa.Table.from_batches(parts).rename_columns(["body_pre", "body_post", "syn_count"]).to_pandas()
    if w.duplicated(["body_pre", "body_post"]).any():
        w = w.groupby(["body_pre", "body_post"], sort=False, as_index=False)["syn_count"].sum()
    return w.reset_index(drop=True)


def load_malecns(data_dir: str | Path = "data", cache: bool = True, min_syn: int = 3,
                 sign_map: dict[str, float] | None = None) -> Connectome:
    """Parse MaleCNS feather files. `min_syn` drops neuron pairs with fewer synapses."""
    data_dir = Path(data_dir)
    cache_path = data_dir / "cache" / f"malecns_min{min_syn}.pt"
    if cache and cache_path.exists():
        log.info("loading cached connectome from %s", cache_path)
        return Connectome.load(cache_path)
    sign_map = sign_map or DEFAULT_SIGN

    neurons = read_neuron_table(data_dir)
    w = read_weights(data_dir, keep_ids=neurons["root_id"].to_numpy(), min_syn=min_syn)
    lut = pd.Series(np.arange(len(neurons)), index=neurons["root_id"].to_numpy())
    log.info("weights table: %s neuron->neuron pairs with >= %d synapses", f"{len(w):,}", min_syn)
    pre = lut.loc[w["body_pre"].to_numpy()].to_numpy()
    post = lut.loc[w["body_post"].to_numpy()].to_numpy()

    nt_pre = neurons["nt_full"].to_numpy()[pre]
    unknown = ~np.isin(nt_pre, list(sign_map))
    if unknown.any():
        log.warning("%s edges (%.1f%%) from neurons with unclear transmitter treated as excitatory",
                    f"{int(unknown.sum()):,}", 100 * unknown.mean())
    sign = np.where(unknown, 1.0, [sign_map.get(t, 1.0) for t in nt_pre]).astype(np.float32)

    c = build_connectome(neurons, pre, post, w["syn_count"].to_numpy().astype(np.float32), sign)
    log.info(c.summary())
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        c.save(cache_path)
    return c
