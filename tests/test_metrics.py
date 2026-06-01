from src.training.metrics import save_loss_plot


def test_save_loss_plot_writes_file(tmp_path):
    plot_path = tmp_path / "loss.pdf"
    save_loss_plot([1.0, 0.8], [1.1, 0.9], examples_seen=1000, save_path=str(plot_path))
    assert plot_path.exists()


def test_save_loss_plot_skips_when_empty(caplog):
    save_loss_plot([], [], examples_seen=0)
    assert "skipping loss plot" in caplog.text.lower()
