"""BridgeClock follows the sim time when the sim runs slower than real time (plan section 19)."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sar.mission.vehicle import BridgeClock


def _fake_sim(rate):
    t0 = time.monotonic()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            b = json.dumps({"sim_time": 100.0 + (time.monotonic() - t0) * rate}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_bridge_clock_follows_slow_sim():
    srv = _fake_sim(0.25)
    clock = BridgeClock(f"http://127.0.0.1:{srv.server_address[1]}", poll=0.05)
    assert clock.now() < 0.5
    w0, m0 = time.monotonic(), clock.now()
    asyncio.run(clock.sleep(0.5))
    wall = time.monotonic() - w0
    # 0.5 s of sim time at a quarter of real time takes about 2 s of wall time
    assert 1.5 < wall < 3.0
    assert clock.now() - m0 >= 0.5
    srv.shutdown()


def test_bridge_clock_monotonic():
    srv = _fake_sim(1.0)
    clock = BridgeClock(f"http://127.0.0.1:{srv.server_address[1]}", poll=0.02)
    last = clock.now()
    for _ in range(200):
        t = clock.now()
        assert t >= last
        last = t
        time.sleep(0.002)
    srv.shutdown()
