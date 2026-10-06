import pytest

torch = pytest.importorskip("torch")
tg = pytest.importorskip("torch_geometric")

from vascular_ged.training import make_model


def test_siamese_symmetry_self_distance_and_reload(tmp_path):
    model = make_model(layers=2, hidden=8, embedding=8).eval()
    a = tg.data.Batch.from_data_list([tg.data.Data(x=torch.tensor([[0., 0., 0.], [1., 0., 0.]]), edge_index=torch.tensor([[0, 1], [1, 0]]))])
    b = tg.data.Batch.from_data_list([tg.data.Data(x=torch.tensor([[0., 0., 0.]]), edge_index=torch.empty((2, 0), dtype=torch.long))])
    assert torch.allclose(model(a, b), model(b, a), atol=1e-7)
    assert torch.allclose(model(a, a), torch.zeros(1), atol=1e-7)
    path = tmp_path / "model.pt"; torch.save(model.state_dict(), path)
    loaded = make_model(layers=2, hidden=8, embedding=8); loaded.load_state_dict(torch.load(path, weights_only=True)); loaded.eval()
    assert torch.allclose(model(a, b), loaded(a, b))
