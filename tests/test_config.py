from srp.config import load_config


def test_config_loads_and_is_sane():
    cfg = load_config()
    assert cfg["walkforward"]["purge_days"] >= cfg["label"]["horizon"]
    assert len(cfg["data"]["candidates"]) >= cfg["data"]["universe_size"]
    assert cfg["montecarlo"]["n_boot"] >= 5000 and cfg["montecarlo"]["n_perm"] >= 1000
