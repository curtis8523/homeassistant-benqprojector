"""Tests for the vendored benqprojector library against a fake projector."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "custom_components" / "benqprojector"))

# pylint: disable=wrong-import-position
from benqlib import benqprojector as lib  # noqa: E402
from benqlib import BenQProjector, BenQProjectorTelnet  # noqa: E402

from .fake_projector import FakeProjector  # noqa: E402


@pytest.fixture(autouse=True)
def fast_reconnect(monkeypatch):
    monkeypatch.setattr(lib, "_RECONNECT_DELAY_MIN", 0.2)
    monkeypatch.setattr(lib, "_RECONNECT_DELAY_MAX", 0.4)


@pytest.fixture
async def fake():
    projector = FakeProjector()
    await projector.start()
    yield projector
    await projector.stop()


async def wait_for(condition, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.05)


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def __call__(self, command, data):
        self.events.append((command, data))

    def last(self, command):
        for name, data in reversed(self.events):
            if name == command:
                return data
        return None


async def test_connect_detects_model_and_mac(fake):
    projector = BenQProjectorTelnet("127.0.0.1", fake.port)
    assert await projector.connect()
    assert projector.model == "LK935"
    assert projector.unique_id == "c0:c4:f9:18:a6:2f"
    assert projector.power_status == BenQProjector.POWERSTATUS_OFF
    await projector.disconnect()


async def test_connect_returns_false_instead_of_assert(fake):
    """Upstream raised AssertionError, which made HA give up on the entry."""
    fake.silent.add("modelname")
    projector = BenQProjectorTelnet("127.0.0.1", fake.port)
    assert await projector.connect() is False
    assert not projector.connected()


async def test_connect_uses_model_hint_when_modelname_fails(fake):
    fake.silent.add("modelname")
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    assert await projector.connect()
    assert projector.model == "LK935"
    await projector.disconnect()


async def test_connect_returns_false_on_reset(fake):
    fake.refuse = True
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    assert await projector.connect() is False


async def test_connect_returns_false_when_nothing_listens():
    projector = BenQProjectorTelnet("127.0.0.1", 1, "LK935")
    assert await projector.connect() is False


async def test_read_loop_recovers_after_reset(fake):
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    events = Recorder()
    assert await projector.connect(interval=0.1)
    projector.add_listener(events)
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_OFF)

    fake.reset_connections()
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_UNKNOWN)

    fake.power = "on"
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_ON)
    assert projector.connected()
    await projector.disconnect()


async def test_read_loop_recovers_after_outage(fake):
    port = fake.port
    projector = BenQProjectorTelnet("127.0.0.1", port, "LK935")
    events = Recorder()
    assert await projector.connect(interval=0.1)
    projector.add_listener(events)
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_OFF)

    await fake.stop()
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_UNKNOWN)
    await asyncio.sleep(0.5)

    await fake.start(port)
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_OFF)
    await projector.disconnect()


async def test_read_loop_survives_unexpected_error(fake, monkeypatch):
    fake.power = "on"
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    events = Recorder()
    assert await projector.connect(interval=0.1)

    calls = 0
    original = projector.update_video_source

    async def broken():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("boom")
        return await original()

    monkeypatch.setattr(projector, "update_video_source", broken)
    projector.add_listener(events)
    await wait_for(lambda: calls >= 3)
    assert projector._read_task is not None and not projector._read_task.done()
    await wait_for(lambda: events.last("sour") == "hdmi")
    await projector.disconnect()


async def test_disconnect_stops_read_task_while_disconnected(fake):
    """Upstream left the task running when the connection was down at unload."""
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    events = Recorder()
    assert await projector.connect(interval=0.1)
    projector.add_listener(events)
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_OFF)
    task = projector._read_task

    fake.refuse = True
    fake.reset_connections()
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_UNKNOWN)
    assert not projector.connected()

    await projector.disconnect()
    assert task.done()

    fake.refuse = False
    connections = fake.connections
    await asyncio.sleep(1)
    assert fake.connections == connections


async def test_unsupported_commands_stop_being_polled(fake):
    fake.power = "on"
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    assert await projector.connect(interval=0.05)
    projector.add_listener(command="micvol")
    projector.add_listener(command="bri")
    events = Recorder()
    projector.add_listener(events)
    await wait_for(lambda: fake.count("bri") >= 10)
    assert fake.count("micvol") == lib._MAX_UNSUPPORTED_RESPONSES
    assert events.last("bri") == "50"
    await projector.disconnect()


async def test_user_command_waits_for_slow_poll(fake):
    """A command issued during a slow poll must not fail with Too busy."""
    fake.power = "on"
    fake.delays["bri"] = 1.5
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    assert await projector.connect(interval=0.05)
    projector.add_listener(command="bri")
    projector.add_listener(Recorder())
    await wait_for(lambda: fake.count("bri") >= 1)

    assert await projector.turn_off()
    assert fake.power == "off"
    await projector.disconnect()


async def test_volume_up_without_known_volume(fake):
    fake.power = "on"
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    assert await projector.connect()
    projector.volume = None
    assert await projector.volume_up()
    assert projector.volume == 6
    await projector.disconnect()


async def test_blocked_power_keeps_state(fake):
    fake.power = "on"
    projector = BenQProjectorTelnet("127.0.0.1", fake.port, "LK935")
    events = Recorder()
    assert await projector.connect(interval=0.1)
    projector.add_listener(events)
    await wait_for(lambda: events.last("pow") == BenQProjector.POWERSTATUS_ON)

    fake.blocked.add("pow")
    count = fake.count("pow")
    await wait_for(lambda: fake.count("pow") >= count + 5)
    assert events.last("pow") == BenQProjector.POWERSTATUS_ON
    await projector.disconnect()