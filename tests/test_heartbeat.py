import json

from pxdesign.utils.heartbeat import HeartbeatReporter


def _write_rank_status(
    output_dir,
    rank,
    *,
    stage="evaluation",
    task_name="task-a",
    seed=123,
    global_run=0,
    produced_total=0,
    expected_total=100,
    primary_counter="eval_designs",
    eval_extra=None,
):
    payload = {
        "pipeline": {
            "stage": stage,
            "task_name": task_name,
            "seed": seed,
            "global_run": global_run,
        },
        "status": {"updated_at": "2026-01-01T00:00:00Z"},
        "progress": {
            "produced_total": produced_total,
            "expected_total": expected_total,
            "primary_counter": primary_counter,
        },
    }
    if eval_extra is not None:
        payload["extra"] = {"eval": eval_extra}
    (output_dir / f"status_rank{rank}.json").write_text(json.dumps(payload))
    return payload


def _write_global_status(output_dir, payload):
    (output_dir / "status.json").write_text(json.dumps(payload))


def _eval_tool_progress():
    return {
        "tool_progress": {
            "af2_complex": {"enabled": True, "done": 10, "total": 10},
            "af2_monomer": {"enabled": False, "done": 0, "total": 0},
            "ptx_mini": {"enabled": True, "done": 10, "total": 10},
            "ptx": {"enabled": False, "done": 0, "total": 0},
        }
    }


def test_aggregate_merges_eval_tool_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "evaluation")
    hb = HeartbeatReporter(tmp_path)
    hb.world_size = 2

    _write_rank_status(
        tmp_path,
        0,
        eval_extra={
            "task": "task-a",
            "step": "run_task",
            "expected_outputs": {"num_seqs": 2, "model_ids": [0, 1, 2]},
            "tool_progress": {
                "af2_complex": {"enabled": True, "done": 30, "total": 50},
                "af2_monomer": {"enabled": False, "done": 50, "total": 50},
                "ptx_mini": {"enabled": True, "done": 0, "total": 50},
                "ptx": {"enabled": False, "done": 50, "total": 50},
            },
        },
    )
    _write_rank_status(
        tmp_path,
        1,
        eval_extra={
            "task": "task-a",
            "step": "run_task",
            "tool_progress": {
                "af2_complex": {"enabled": True, "done": 20, "total": 50},
                "af2_monomer": {"enabled": False, "done": 50, "total": 50},
                "ptx_mini": {"enabled": True, "done": 0, "total": 50},
                "ptx": {"enabled": False, "done": 50, "total": 50},
            },
        },
    )

    agg = hb._aggregate(now=0)
    eval_extra = agg["extra"]["eval"]

    assert eval_extra["task"] == "task-a"
    assert eval_extra["step"] == "run_task"
    assert eval_extra["expected_outputs"] == {"num_seqs": 2, "model_ids": [0, 1, 2]}
    assert eval_extra["active_tool"] == "af2_complex"
    assert eval_extra["active_group"] == "af2_eval"
    assert eval_extra["tool_progress"]["af2_complex"] == {
        "enabled": True,
        "done": 50,
        "total": 100,
    }
    assert eval_extra["tool_progress"]["af2_monomer"] == {
        "enabled": False,
        "done": 0,
        "total": 0,
    }


def test_aggregate_does_not_carry_eval_extra_outside_eval_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "ranking")
    hb = HeartbeatReporter(tmp_path)
    hb.world_size = 1
    _write_rank_status(
        tmp_path,
        0,
        stage="ranking",
        eval_extra={
            "tool_progress": {
                "af2_complex": {"enabled": True, "done": 100, "total": 100},
                "af2_monomer": {"enabled": False, "done": 0, "total": 0},
                "ptx_mini": {"enabled": True, "done": 100, "total": 100},
                "ptx": {"enabled": False, "done": 0, "total": 0},
            }
        },
    )

    agg = hb._aggregate(now=0)
    assert "extra" not in agg


def test_same_stage_counter_regression_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "diffusion")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    _write_rank_status(
        tmp_path,
        0,
        stage="diffusion",
        produced_total=10,
        expected_total=10,
        primary_counter="diffusion_samples",
    )

    hb.update(
        produced_total=0,
        expected_total=10,
        primary_counter="diffusion_samples",
        force=True,
    )

    status = json.loads((tmp_path / "status_rank0.json").read_text())
    assert status["progress"]["produced_total"] == 10
    assert status["progress"]["expected_total"] == 10


def test_same_run_stage_regression_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "evaluation")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    _write_rank_status(
        tmp_path,
        0,
        stage="ranking",
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
    )

    hb.update(
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
        force=True,
    )

    status = json.loads((tmp_path / "status_rank0.json").read_text())
    assert status["pipeline"]["stage"] == "ranking"


def test_incoherent_evaluation_diffusion_counter_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "evaluation")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    hb.touch(force=True)

    assert not (tmp_path / "status_rank0.json").exists()
    assert not (tmp_path / "status.json").exists()


def test_eval_tool_progress_rank_status_is_not_erased(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "evaluation")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    _write_rank_status(
        tmp_path,
        0,
        stage="evaluation",
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
        eval_extra=_eval_tool_progress(),
    )

    hb.update(
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
        force=True,
    )

    status = json.loads((tmp_path / "status_rank0.json").read_text())
    assert status["extra"]["eval"]["tool_progress"]["af2_complex"] == {
        "enabled": True,
        "done": 10,
        "total": 10,
    }


def test_eval_tool_progress_aggregate_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "evaluation")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    existing_global = _write_rank_status(
        tmp_path,
        99,
        stage="evaluation",
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
        eval_extra=_eval_tool_progress(),
    )
    _write_global_status(tmp_path, existing_global)
    (tmp_path / "status_rank99.json").unlink()

    hb.update(
        produced_total=10,
        expected_total=10,
        primary_counter="eval_designs",
        force=True,
    )

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["extra"]["eval"]["tool_progress"]["ptx_mini"] == {
        "enabled": True,
        "done": 10,
        "total": 10,
    }


def test_expected_total_zero_is_preserved_in_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("PXDESIGN_STAGE", "diffusion")
    monkeypatch.setenv("PXDESIGN_TASK_NAME", "task-a")
    monkeypatch.setenv("PXDESIGN_SEED", "123")
    monkeypatch.setenv("PXDESIGN_GLOBAL_RUN", "0")
    hb = HeartbeatReporter(tmp_path)
    hb.rank = 0
    hb.world_size = 1

    hb.update(
        produced_total=0,
        expected_total=0,
        primary_counter="diffusion_samples",
        force=True,
    )

    status = json.loads((tmp_path / "status_rank0.json").read_text())
    assert status["progress"]["expected_total"] == 0
    assert status["progress"]["produced_total"] == 0
