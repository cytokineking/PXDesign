import json

from pxdesign.utils.heartbeat import HeartbeatReporter


def _write_rank_status(output_dir, rank, *, stage="evaluation", eval_extra=None):
    payload = {
        "pipeline": {"stage": stage, "task_name": "task-a"},
        "status": {"updated_at": "2026-01-01T00:00:00Z"},
        "progress": {"produced_total": 0, "expected_total": 100},
    }
    if eval_extra is not None:
        payload["extra"] = {"eval": eval_extra}
    (output_dir / f"status_rank{rank}.json").write_text(json.dumps(payload))


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
