from __future__ import annotations

import copy

import pytest

from src.common.config import ConfigError, apply_overrides, load_config, validate


def _bad(cfg, mutate):
    raw = copy.deepcopy(cfg.raw)
    mutate(raw)
    with pytest.raises(ConfigError):
        validate(raw)


def test_valid_config_loads(cfg):
    assert cfg["grid_res"] == 0.25
    assert cfg.get("train.lr") == pytest.approx(3e-4)
    assert cfg.get("nope.nothing", 5) == 5
    with pytest.raises(KeyError):
        cfg.get("nope.nothing")
    assert len(cfg.hash()) == 64


def test_hash_changes_with_content(cfg):
    other = apply_overrides(cfg.raw, ["train.lr=1.0e-3"])
    assert cfg.hash() != type(cfg)(raw=other, path=cfg.path).hash()


def test_dump_roundtrip(cfg, tmp_path):
    cfg.dump(tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml").hash() == cfg.hash()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("seed"),
        lambda r: r["region"].update(lat_max=4.0),
        lambda r: r["region"].update(lat_max=23.1),
        lambda r: r["dates"].update(start="2023-01-01"),
        lambda r: r["splits"].update(val=[2020, 2021]),
        lambda r: r["splits"].update(test=[2022, 2025]),
        lambda r: r.update(depths_m=[0, 20, 10] + r["depths_m"][3:]),
        lambda r: r.update(depths_m=[5] + r["depths_m"][1:]),
        lambda r: r["depth_bands_m"].update({"x": [12345]}),
        lambda r: r["depth_bands_m"]["0-100"].append(125),
        lambda r: r["inputs"].append("foo"),
        lambda r: r["channels"]["sst"].update(product="nonexistent"),
        lambda r: r["model"].update(patch=6),
        lambda r: r["model"].update(heads=7),
        lambda r: r["train"].update(lr="3e-4"),
        lambda r: r["train"].update(depth_weights="bad"),
        lambda r: r["train"].update(target_mode="bad"),
        lambda r: r["train"].update(amp="bad"),
        lambda r: r["eval"]["seasons"]["pre_monsoon"].append(1),
        lambda r: r["train"].update(drop_inputs=["nonexistent"]),
        lambda r: r["train"].update(patience=1000),
    ],
)
def test_each_rule_triggers(cfg, mutate):
    _bad(cfg, mutate)


def test_all_errors_reported_together(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["train"]["lr"] = "3e-4"
    raw["model"]["patch"] = 6
    with pytest.raises(ConfigError) as e:
        validate(raw)
    assert "train.lr" in str(e.value) and "model.patch" in str(e.value)


def test_yaml_string_lr_rejected(cfg, tmp_path):
    text = (cfg.path).read_text(encoding="utf-8").replace("3.0e-4", "3e-4")
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_overrides_typed(cfg):
    out = apply_overrides(
        cfg.raw,
        ["train.lr=1.0e-4", "train.epochs=3", "train.smoke=true", "train.drop_inputs=[sss]",
         "train.new_flag=7", "extra_inputs.land_mask=true", "model.name=cnn_unet"],
    )  # fmt: skip
    assert out["train"]["lr"] == 1e-4 and isinstance(out["train"]["lr"], float)
    assert out["train"]["epochs"] == 3 and out["train"]["smoke"] is True
    assert out["train"]["drop_inputs"] == ["sss"]
    assert out["train"]["new_flag"] == 7
    assert out["extra_inputs"]["land_mask"] is True
    assert out["model"]["name"] == "cnn_unet"
    assert "new_flag" not in cfg["train"]  # original untouched


@pytest.mark.parametrize("bad", ["newtop=1", "model.newkey=1", "noequals", "nope.x=1"])
def test_bad_overrides(cfg, bad):
    with pytest.raises(ConfigError):
        apply_overrides(cfg.raw, [bad])
