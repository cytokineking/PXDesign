import importlib
import json
import sys
import types

import pytest


def _install_pipeline_import_stubs():
    sys.modules.setdefault("torch", types.ModuleType("torch"))

    protenix_config = types.ModuleType("protenix.config")
    protenix_config.save_config = lambda *args, **kwargs: None
    protenix_utils = types.ModuleType("protenix.utils")
    protenix_dist = types.ModuleType("protenix.utils.distributed")

    class _Dist:
        rank = 0
        world_size = 1
        local_rank = 0

        def barrier(self):
            return None

    protenix_dist.DIST_WRAPPER = _Dist()
    protenix_seed = types.ModuleType("protenix.utils.seed")
    protenix_seed.seed_everything = lambda *args, **kwargs: None
    sys.modules.setdefault("protenix", types.ModuleType("protenix"))
    sys.modules["protenix.config"] = protenix_config
    sys.modules["protenix.utils"] = protenix_utils
    sys.modules["protenix.utils.distributed"] = protenix_dist
    sys.modules["protenix.utils.seed"] = protenix_seed

    pxdbench_aggregate = types.ModuleType("pxdbench.aggregate")
    pxdbench_aggregate.aggregate_binder_eval = lambda *args, **kwargs: None
    pxdbench_run = types.ModuleType("pxdbench.run")
    pxdbench_run.run_task = lambda *args, **kwargs: None
    pxdbench_utils = types.ModuleType("pxdbench.utils")
    pxdbench_utils.convert_cif_to_pdb = lambda *args, **kwargs: None
    pxdbench_utils.convert_cifs_to_pdbs = lambda *args, **kwargs: (None, [], None, None)
    pxdbench_utils.find_binder_chains = lambda *args, **kwargs: []
    pxdbench_utils.find_cond_chains = lambda *args, **kwargs: []
    pxdbench_utils.str2bool = lambda value: str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    sys.modules.setdefault("pxdbench", types.ModuleType("pxdbench"))
    sys.modules["pxdbench.aggregate"] = pxdbench_aggregate
    sys.modules["pxdbench.run"] = pxdbench_run
    sys.modules["pxdbench.utils"] = pxdbench_utils

    helpers = types.ModuleType("pxdesign.runner.helpers")
    helpers.save_top_designs = lambda *args, **kwargs: None
    helpers.use_target_template_or_not = lambda *args, **kwargs: False
    inference = types.ModuleType("pxdesign.runner.inference")
    inference.InferenceRunner = object
    presets = types.ModuleType("pxdesign.runner.presets")
    presets.PRESETS = {}
    infer = types.ModuleType("pxdesign.utils.infer")
    infer.convert_to_bioassembly_dict = lambda *args, **kwargs: {}
    infer.derive_seed = lambda *args, **kwargs: 0
    infer.download_inference_cache = lambda *args, **kwargs: None
    infer.get_configs = lambda *args, **kwargs: None
    inputs = types.ModuleType("pxdesign.utils.inputs")
    inputs.process_input_file = lambda *args, **kwargs: None
    pipeline_utils = types.ModuleType("pxdesign.utils.pipeline")
    pipeline_utils.check_tool_weights = lambda *args, **kwargs: None
    sys.modules["pxdesign.runner.helpers"] = helpers
    sys.modules["pxdesign.runner.inference"] = inference
    sys.modules["pxdesign.runner.presets"] = presets
    sys.modules["pxdesign.utils.infer"] = infer
    sys.modules["pxdesign.utils.inputs"] = inputs
    sys.modules["pxdesign.utils.pipeline"] = pipeline_utils


_install_pipeline_import_stubs()
pipeline = importlib.import_module("pxdesign.runner.pipeline")


class _EvalCfg:
    eval_complex = True
    eval_binder_monomer = True
    eval_protenix = True
    eval_protenix_mini = False
    num_seqs = 1

    class tools:
        class af2:
            model_ids = [0]


class _EvalCfgTwoAf2Models(_EvalCfg):
    class tools:
        class af2:
            model_ids = [0, 1]


def _write_marker(root, marker):
    marker_dir = root / "output" / ".aggregation_seed"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / "complete.json").write_text(json.dumps(marker))


def _base_marker(**overrides):
    marker = {
        "validated": True,
        "run_dir": "runs/run_000",
        "tasks": ["task-a"],
        "enabled_tools": {
            "af2_complex": True,
            "af2_monomer": True,
            "ptx": True,
            "ptx_mini": False,
        },
        "expected_counts": {
            "diffusion_cif_count": 10,
            "sequence_txt_count": 10,
            "af2_count": 20,
            "ptx_count": 10,
            "ptx_mini_count": 0,
        },
        "counts": {
            "diffusion_cif_count": 10,
            "sequence_txt_count": 10,
            "af2_count": 20,
            "ptx_count": 10,
            "ptx_mini_count": 0,
        },
    }
    for key, value in overrides.items():
        marker[key] = value
    return marker


def _ariax_split_marker():
    marker = _base_marker()
    split_counts = {
        "diffusion_cif": 10,
        "seq_txt": 10,
        "af2_json": 20,
        "af2_pdb": 20,
        "ptx_json": 10,
        "ptx_pdb": 10,
        "ptx_mini": 0,
    }
    marker["expected_counts"] = dict(split_counts)
    marker["counts"] = dict(split_counts)
    return marker


_DEFAULT_PDB_NAMES = object()


def _marker_status(
    tmp_path,
    monkeypatch,
    marker,
    *,
    eval_cfg=None,
    pdb_names=_DEFAULT_PDB_NAMES,
):
    _write_marker(tmp_path, marker)
    monkeypatch.setattr(pipeline, "_rw_project_root_for_path", lambda path: str(tmp_path))
    if pdb_names is _DEFAULT_PDB_NAMES:
        pdb_names = [f"task-a_sample_{i:06d}" for i in range(10)]
    return pipeline._aggregation_seed_marker_status(
        "/root/pxdesign-work/project/output/runs/run_000/eval/task-a",
        run_dir="runs/run_000",
        task_name="task-a",
        eval_cfg=eval_cfg or _EvalCfg(),
        expected_total=10,
        pdb_names=pdb_names,
        run_seed=123,
    )


def test_legacy_count_marker_bypasses_scans(tmp_path, monkeypatch):
    status = _marker_status(tmp_path, monkeypatch, _base_marker())

    assert status["valid"] is True
    assert status["usable_for_legacy_scan_bypass"] is True
    assert status["usable_for_completeness"] is False
    assert status["mode"] == "legacy_counts_scan_bypass"
    assert status["counts"]["af2_total"]["expected"] == 20


def test_legacy_count_marker_rejects_enabled_tool_mismatch(tmp_path, monkeypatch):
    marker = _base_marker(
        enabled_tools={
            "af2_complex": True,
            "af2_monomer": True,
            "ptx": True,
            "ptx_mini": True,
        }
    )
    status = _marker_status(tmp_path, monkeypatch, marker)

    assert status["usable_for_legacy_scan_bypass"] is False
    assert status["reason"].startswith("enabled_tool_mismatch:ptx_mini")


def test_legacy_count_marker_rejects_expected_count_mismatch(tmp_path, monkeypatch):
    marker = _base_marker()
    marker["expected_counts"]["af2_count"] = 10
    status = _marker_status(tmp_path, monkeypatch, marker)

    assert status["usable_for_legacy_scan_bypass"] is False
    assert status["reason"].startswith("expected_count_mismatch:af2_total")


def test_legacy_count_marker_accepts_ariax_split_count_keys(tmp_path, monkeypatch):
    status = _marker_status(tmp_path, monkeypatch, _ariax_split_marker())

    assert status["valid"] is True
    assert status["usable_for_legacy_scan_bypass"] is True
    assert status["mode"] == "legacy_counts_scan_bypass"
    assert status["counts"]["diffusion_cif"]["observed"] == 10
    assert status["counts"]["seq_txt"]["observed"] == 10
    assert status["counts"]["af2_json"]["observed"] == 20
    assert status["counts"]["af2_pdb"]["observed"] == 20
    assert status["counts"]["ptx_json"]["observed"] == 10
    assert status["counts"]["ptx_pdb"]["observed"] == 10


def test_split_observed_counts_accept_combined_expected_counts(tmp_path, monkeypatch):
    marker = _ariax_split_marker()
    marker["expected_counts"] = {
        "diffusion_cif": 10,
        "seq_txt": 10,
        "af2_count": 20,
        "ptx_count": 10,
        "ptx_mini": 0,
    }
    status = _marker_status(tmp_path, monkeypatch, marker)

    assert status["usable_for_legacy_scan_bypass"] is True
    assert status["counts"]["af2_json"]["marker_expected"] == 20
    assert status["counts"]["af2_pdb"]["marker_expected"] == 20
    assert status["counts"]["ptx_json"]["marker_expected"] == 10
    assert status["counts"]["ptx_pdb"]["marker_expected"] == 10


def test_split_count_marker_requires_af2_json_and_pdb(tmp_path, monkeypatch):
    marker = _ariax_split_marker()
    marker["counts"]["af2_pdb"] = 19
    status = _marker_status(tmp_path, monkeypatch, marker)

    assert status["usable_for_legacy_scan_bypass"] is False
    assert status["reason"].startswith("observed_count_insufficient:af2_pdb")


def test_af2_expected_count_includes_model_ids(tmp_path, monkeypatch):
    marker = _ariax_split_marker()
    marker["expected_counts"]["af2_json"] = 40
    marker["expected_counts"]["af2_pdb"] = 40
    marker["counts"]["af2_json"] = 40
    marker["counts"]["af2_pdb"] = 40
    status = _marker_status(
        tmp_path,
        monkeypatch,
        marker,
        eval_cfg=_EvalCfgTwoAf2Models(),
    )

    assert status["usable_for_legacy_scan_bypass"] is True
    assert status["counts"]["af2_json"]["expected"] == 40
    assert status["counts"]["af2_pdb"]["expected"] == 40


def test_af2_model_id_mismatch_rejects_legacy_bypass(tmp_path, monkeypatch):
    status = _marker_status(
        tmp_path,
        monkeypatch,
        _ariax_split_marker(),
        eval_cfg=_EvalCfgTwoAf2Models(),
    )

    assert status["usable_for_legacy_scan_bypass"] is False
    assert status["reason"].startswith("expected_count_mismatch:af2_json")


def test_legacy_count_marker_does_not_require_prescanned_pdb_names(tmp_path, monkeypatch):
    status = _marker_status(
        tmp_path,
        monkeypatch,
        _ariax_split_marker(),
        pdb_names=None,
    )

    assert status["usable_for_legacy_scan_bypass"] is True
    assert status["current_pdb_names_digest"] == ""
    assert status["reason"].startswith("legacy_counts_complete")


def test_marker_complete_heartbeat_is_scan_free(monkeypatch):
    class _Hb:
        payload = None

        def update(self, **kwargs):
            self.payload = kwargs

    def _should_not_scan(*args, **kwargs):
        raise AssertionError("marker-complete heartbeat should not scan")

    monkeypatch.setattr(pipeline, "_pending_pdb_names", _should_not_scan)
    hb = _Hb()
    pipeline._update_eval_heartbeat(
        hb,
        task_name="task-a",
        task_eval_dir="/tmp/eval",
        pdb_names=["task-a_sample_000000", "task-a_sample_000001"],
        eval_cfg=_EvalCfg(),
        seed=123,
        step="aggregate",
        scan_complete=True,
        marker_status={
            "valid": True,
            "mode": "legacy_counts_scan_bypass",
            "reason": "legacy_counts_complete",
        },
    )

    assert hb.payload["produced_total"] == 2
    eval_extra = hb.payload["extra"]["eval"]
    assert eval_extra["owned_pending"] == 0
    assert eval_extra["tool_progress"]["af2_complex"] == {
        "enabled": True,
        "done": 2,
        "total": 2,
    }
    assert eval_extra["tool_progress"]["ptx"] == {
        "enabled": True,
        "done": 2,
        "total": 2,
    }
