import pandas as pd
import torch

from nfly import ConnectomeRNN, InputDrive, load_malecns, stimulate
from nfly.connectome import write_synthetic
from nfly.connectome.malecns import DEFAULT_SIGN, flow_of


def make(tmp_path, **kw):
    return load_malecns(write_synthetic(tmp_path / "d", **kw), cache=False)


def test_load_shapes_signs_and_flow(tmp_path):
    c = make(tmp_path)
    assert c.n_neurons == 2 * 30 * 3 + 300 + 20 + 10            # unannotated fragment excluded
    assert c.pre.shape == c.post.shape == c.sign.shape == c.weight.shape
    nt = c.neurons["nt_full"].to_numpy()[c.pre.numpy()]
    assert torch.equal(c.sign, torch.tensor([DEFAULT_SIGN.get(t, 1.0) for t in nt]))
    assert c.neurons["flow"].value_counts().to_dict() == {"intrinsic": 2 * 30 * 2 + 300 + 20, "afferent": 60, "efferent": 10}
    assert (c.neurons.loc[c.neurons.cell_type == "L1", "hex1"] > 0).all()


def test_flow_mapping():
    sc = pd.Series(["ol_sensory", "descending_neuron", "vnc_motor", "cb_intrinsic", "cb_endocrine", "sensory_ascending"])
    assert flow_of(sc).tolist() == ["afferent", "intrinsic", "efferent", "intrinsic", "efferent", "afferent"]


def test_min_syn_and_duplicate_pairs(tmp_path):
    ann = pd.DataFrame({"bodyId": [1, 2, 3, 4], "superclass": ["ol_sensory", "cb_intrinsic", "vnc_motor", None],
                        "type": ["R1", "X", "MN", None], "class": ["visual", None, None, None], "somaSide": ["L", "R", "L", None]})
    nt = pd.DataFrame({"body": [1, 2, 3], "consensus_nt": ["histamine", "gaba", "unclear"], "predicted_nt_confidence": [0.9, 0.8, 0.1]})
    w = pd.DataFrame({"body_pre": [1, 1, 2, 4, 2], "body_post": [2, 2, 3, 3, 1], "weight": [10, 5, 7, 99, 1]})
    ann.to_feather(tmp_path / "body-annotations.feather"); nt.to_feather(tmp_path / "body-neurotransmitters.feather")
    w.to_feather(tmp_path / "connectome-weights.feather")
    c = load_malecns(tmp_path, cache=False, min_syn=2)
    assert c.n_neurons == 3 and c.n_edges == 2           # body 4 not a neuron; 1-synapse edge dropped
    assert c.syn_count.tolist() == [15.0, 7.0] and c.sign.tolist() == [-1.0, -1.0]
    assert c.neurons["flow"].tolist() == ["afferent", "intrinsic", "efferent"]


def test_post_normalisation(tmp_path):
    c = make(tmp_path)
    total = torch.zeros(c.n_neurons).index_add_(0, c.post, c.weight)
    has_input = torch.zeros(c.n_neurons, dtype=torch.bool); has_input[c.post] = True
    assert torch.allclose(total[has_input], torch.ones(int(has_input.sum())), atol=1e-5)


def test_recurrent_input_matches_dense(tmp_path):
    c = make(tmp_path, n_columns=5, n_central=40, edges=300)
    m = ConnectomeRNN(c); w = m.edge_weights()
    dense = torch.zeros(c.n_neurons, c.n_neurons).index_put_((c.post, c.pre), w, accumulate=True)
    h = torch.randn(3, c.n_neurons)
    assert torch.allclose(m.recurrent_input(h, w), h @ dense.T, atol=1e-5)


def test_forward_and_gradients(tmp_path):
    c = make(tmp_path, n_columns=5, n_central=60, edges=500)
    m = ConnectomeRNN(c, alpha_init=0.3)
    out = m(InputDrive.constant(c.input_nodes(), 1.0, steps=10, batch=2), record=c.output_nodes())
    assert out.shape == (2, 11, len(c.output_nodes()))
    out.pow(2).mean().backward()
    assert m.log_gain.grad is not None and m.bias.grad is not None and m.alpha_logit.grad is not None
    with torch.no_grad():
        m.log_gain.add_(torch.randn_like(m.log_gain))
    assert torch.equal(torch.sign(m.edge_weights()), c.sign)


def test_subset(tmp_path):
    c = make(tmp_path)
    s = c.subset(c.where(flow=["intrinsic", "efferent"]))
    assert (s.neurons["flow"] != "afferent").all() and s.n_edges <= c.n_edges


def test_stimulate_runs(tmp_path):
    c = make(tmp_path)
    res = stimulate(c, ConnectomeRNN(c, global_scale=5.0), c.input_nodes()[:5], steps=8, top=5)
    assert len(res.active_per_step) == 9 and not res.top_neurons["stimulated"].any()
    assert "top downstream" in res.report()


def test_weights_cache_invalidates_after_parameter_update(tmp_path):
    c = make(tmp_path, n_columns=5, n_central=40, edges=300)
    m = ConnectomeRNN(c)
    with torch.no_grad():
        w1 = m.weights()
        assert m.weights() is w1                         # cached while parameters are unchanged
        m.log_gain.add_(1.0)                             # in-place update, as an optimizer does
        w2 = m.weights()
    assert w2 is not w1 and torch.allclose(w2.w, w1.w * torch.e)
    with torch.enable_grad():
        assert m.weights().w.requires_grad               # training path never uses the cache
    with torch.no_grad():
        m.weights()
        m.double()                                       # .to() replaces parameter storage
        assert m.weights().w.dtype == torch.float64      # cache must not hand back the old tensors
