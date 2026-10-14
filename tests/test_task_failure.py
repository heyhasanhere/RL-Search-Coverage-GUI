"""A drone task that raises must not wedge the drone or the mission (Orchestrator._guard)."""

import asyncio

from test_mission import make, run_until, square


async def failing(msg="boom"):
    await asyncio.sleep(0)
    raise RuntimeError(msg)


async def _airborne_search(orch):
    orch.start_mission(polygon=square(), mode="greedy", n_search=1, expected_survivors=1)
    d = orch.drones[0]
    assert await run_until(orch, lambda: d.plan_cell is not None and d.vehicle.tel.in_air, 300)
    return d


def test_airborne_failure_returns_lands_and_mission_finishes():
    async def main():
        orch = make()
        await orch.start()
        d = await _airborne_search(orch)
        orch._set_task(d, failing(), "SEARCH", "x")
        assert await run_until(orch, lambda: d.state == "ABORT", 20)
        assert await run_until(orch, lambda: d.state == "LANDED", 600)
        assert not d.vehicle.tel.in_air
        orch.stop_mission("test")
        assert await run_until(orch, lambda: orch.mission.status == "finished", 900)
        await orch.shutdown()
    asyncio.run(main())


def test_failed_abort_return_becomes_failed_and_mission_finishes():
    async def main():
        orch = make()
        await orch.start()
        d = await _airborne_search(orch)

        async def bad_return(alt):
            raise RuntimeError("return failed")
        d.vehicle.return_home = bad_return
        orch._set_task(d, failing(), "SEARCH", "x")
        assert await run_until(orch, lambda: d.state == "FAILED", 600)
        assert d.vehicle.tel.in_air
        orch.stop_mission("test")
        assert await run_until(orch, lambda: orch.mission.status == "finished", 900)
        assert d.state == "FAILED"
        await orch.shutdown()
    asyncio.run(main())


def test_ground_failure_becomes_landed():
    async def main():
        orch = make()
        await orch.start()
        d = orch.drones[0]
        assert not d.vehicle.tel.in_air
        orch._set_task(d, failing(), "SEARCH", "x")
        assert await run_until(orch, lambda: d.state == "LANDED", 20)
        await orch.shutdown()
    asyncio.run(main())
