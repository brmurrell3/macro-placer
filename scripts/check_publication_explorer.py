#!/usr/bin/env python3
"""Browser QA for the self-contained placement explorer (writeup/interactive/).

Drives an installed Chrome or Chromium headless over the DevTools protocol, using
only the Python standard library. For each viewport (736 and 360 px) and color
scheme (light, dark) it checks that:

- the page opens in the displacement view with all 17 designs;
- every design renders, selects a macro, fits its labels, shows the strict-bounds
  line, and causes no horizontal scroll;
- keyboard focus and Enter switch to the divider view, and End/Home/ArrowRight
  move the divider;
- a pointer click on a moved macro selects it;
- no console error or warning is logged.

It writes per-design states to <qa-dir>/checks.json and screenshots of the
default and divider views to <qa-dir>/. Exit status 0 means every check passed.

Run from the repository root:

    .venv/bin/python scripts/check_publication_explorer.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE = ROOT / 'writeup/interactive'
CHROME_CANDIDATES = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                     '/Applications/Chromium.app/Contents/MacOS/Chromium',
                     'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']
WIDTHS = (736, 360)
SCHEMES = ('light', 'dark')
DESIGNS = 17


class WebSocket:
    """Minimal client-side WebSocket (RFC 6455) for one DevTools target."""

    def __init__(self, url: str, timeout: float = 60):
        assert url.startswith('ws://'), url
        hostport, path = url[5:].split('/', 1)
        host, port = hostport.split(':')
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n'
                           f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            buf += self.sock.recv(4096)
        head, self.buf = buf.split(b'\r\n\r\n', 1)
        if b' 101 ' not in head.split(b'\r\n')[0]:
            raise RuntimeError(f'WebSocket handshake failed: {head[:200]!r}')
        self.next_id = 0
        self.events = []

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise EOFError('DevTools connection closed')
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self) -> dict:
        message = b''
        while True:
            b0, b1 = self._read(2)
            length = b1 & 0x7f
            if length == 126:
                length = struct.unpack('>H', self._read(2))[0]
            elif length == 127:
                length = struct.unpack('>Q', self._read(8))[0]
            payload = self._read(length)
            opcode = b0 & 0x0f
            if opcode == 8:
                raise EOFError('DevTools connection closed')
            if opcode in (0, 1, 2):
                message += payload
                if b0 & 0x80:
                    return json.loads(message)

    def send(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self.next_id += 1
        data = json.dumps({'id': self.next_id, 'method': method, 'params': params or {}}).encode()
        n = len(data)
        if n < 126:
            header = bytes([0x81, 0x80 | n])
        elif n < 65536:
            header = bytes([0x81, 0x80 | 126]) + struct.pack('>H', n)
        else:
            header = bytes([0x81, 0x80 | 127]) + struct.pack('>Q', n)
        mask = os.urandom(4)
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv()
            if msg.get('id') == self.next_id:
                if 'error' in msg:
                    raise RuntimeError(f'{method}: {msg["error"]}')
                return msg.get('result', {})
            self.events.append(msg)
        raise TimeoutError(method)

    def wait_event(self, method: str, timeout: float = 30) -> dict:
        for i, event in enumerate(self.events):
            if event.get('method') == method:
                return self.events.pop(i)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv()
            if msg.get('method') == method:
                return msg
            self.events.append(msg)
        raise TimeoutError(method)


def evaluate(ws: WebSocket, expression: str):
    result = ws.send('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
    if 'exceptionDetails' in result:
        raise RuntimeError(json.dumps(result['exceptionDetails'])[:800])
    return result['result'].get('value')


def wait_for(ws: WebSocket, expression: str, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if evaluate(ws, expression):
            return
        time.sleep(0.1)
    raise TimeoutError(expression)


def key(ws: WebSocket, name: str, code: str, vk: int, text: str | None = None) -> None:
    params = {'type': 'keyDown', 'key': name, 'code': code, 'windowsVirtualKeyCode': vk, 'nativeVirtualKeyCode': vk}
    if text:
        params['text'] = text
    ws.send('Input.dispatchKeyEvent', params)
    ws.send('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': name, 'code': code,
                                       'windowsVirtualKeyCode': vk, 'nativeVirtualKeyCode': vk})


def screenshot(ws: WebSocket, path: Path) -> None:
    box = evaluate(ws, """(() => { const r = document.getElementById('thinkorplace-placement-comparison').getBoundingClientRect();
        return {x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height}; })()""")
    pad = 8
    clip = {'x': max(0, box['x'] - pad), 'y': max(0, box['y'] - pad), 'width': box['width'] + 2 * pad,
            'height': box['height'] + 2 * pad, 'scale': 1}
    shot = ws.send('Page.captureScreenshot', {'format': 'png', 'clip': clip, 'captureBeyondViewport': True})
    path.write_bytes(base64.b64decode(shot['data']))


STATE = """(() => { const root = document.getElementById('thinkorplace-placement-comparison');
  return { fits: root.dataset.labelsFit, scrollWidth: document.documentElement.scrollWidth, viewportWidth: innerWidth,
    count: root.querySelectorAll('rect.tp-macro').length, arrows: root.querySelectorAll('.tp-arrow').length,
    detail: root.querySelector('[data-detail]').textContent, divider: root.querySelector('#tp-divider').value,
    selected: root.querySelectorAll('[data-movement]').length,
    before: root.querySelector('[data-stage=before]').textContent, after: root.querySelector('[data-stage=after]').textContent,
    view: root.querySelector('[aria-pressed=true]').dataset.view }; })()"""


def find_chrome(explicit: str | None) -> str:
    for candidate in ([explicit] if explicit else []) + [os.environ.get('CHROME', '')] + CHROME_CANDIDATES:
        if not candidate:
            continue
        path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if path and os.access(path, os.X_OK):
            return path
    raise SystemExit('No Chrome or Chromium found; pass --chrome or set CHROME.')


def start_chrome(chrome: str, profile: str) -> tuple[subprocess.Popen, WebSocket]:
    proc = subprocess.Popen([chrome, '--headless=new', '--remote-debugging-port=0', f'--user-data-dir={profile}',
                             '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', 'about:blank'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = Path(profile) / 'DevToolsActivePort'
    for _ in range(200):
        if port_file.exists() and port_file.read_text().strip():
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        raise SystemExit('Chrome did not open a DevTools port')
    port = port_file.read_text().split()[0]
    targets = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'))
    ws = WebSocket(next(t for t in targets if t['type'] == 'page')['webSocketDebuggerUrl'])
    for domain in ('Page', 'Runtime', 'Log'):
        ws.send(f'{domain}.enable')
    return proc, ws


def check_view(ws: WebSocket, page: Path, qa: Path, names: list, width: int, scheme: str,
               checks: list, failures: list) -> None:
    ws.events.clear()
    ws.send('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 1000, 'deviceScaleFactor': 2,
                                                   'mobile': width < 400})
    # The narrow view also exercises reduced motion.
    ws.send('Emulation.setEmulatedMedia', {'features': [
        {'name': 'prefers-color-scheme', 'value': scheme},
        {'name': 'prefers-reduced-motion', 'value': 'reduce' if width < 400 else 'no-preference'}]})
    ws.send('Page.navigate', {'url': page.as_uri()})
    ws.wait_event('Page.loadEventFired')
    wait_for(ws, "document.querySelectorAll('rect.tp-macro').length > 0")
    time.sleep(0.3)
    initial = evaluate(ws, STATE)
    if initial['view'] != 'displacement':
        failures.append(('default view', width, scheme, initial['view']))
    options = evaluate(ws, "document.querySelectorAll('[data-benchmark] option').length")
    if options != DESIGNS:
        failures.append(('design count', width, scheme, options))
    screenshot(ws, qa / f'{width}-{scheme}.png')
    background = evaluate(ws, "getComputedStyle(document.body).backgroundColor")
    transition = evaluate(ws, "getComputedStyle(document.querySelector('[data-view]')).transitionDuration")

    # Every design: switch, select the first macro in data order, check layout and the strict-bounds line.
    for name in names:
        evaluate(ws, f"""(() => {{ const s = document.querySelector('[data-benchmark]'); s.value = '{name}';
            s.dispatchEvent(new Event('change')); const m = document.querySelector('[data-macro]'); m.value = '0';
            m.dispatchEvent(new Event('change')); }})()""")
        time.sleep(0.05)
        state = evaluate(ws, STATE)
        ok = (state['scrollWidth'] <= state['viewportWidth'] and state['selected'] == 1
              and state['fits'] == 'true' and 'hard overlap' in state['after'])
        checks.append({'outerViewportWidth': width, 'colorScheme': scheme, 'name': name, **state})
        if not ok:
            failures.append(('layout', width, scheme, name, state))

    # Keyboard: focus the divider-view button, activate it with Enter, then move the divider.
    evaluate(ws, "document.querySelector('[data-benchmark]').value='ibm17'; "
                 "document.querySelector('[data-benchmark]').dispatchEvent(new Event('change'))")
    evaluate(ws, "document.querySelector('[data-view=divider]').focus()")
    key(ws, 'Enter', 'Enter', 13, '\r')
    time.sleep(0.1)
    evaluate(ws, "document.getElementById('tp-divider').focus()")
    key(ws, 'End', 'End', 35)
    after_end = evaluate(ws, STATE)
    key(ws, 'Home', 'Home', 36)
    key(ws, 'ArrowRight', 'ArrowRight', 39)
    after_home = evaluate(ws, STATE)
    keyboard_ok = after_end['view'] == 'divider' and after_end['divider'] == '100' and after_home['divider'] == '1'
    if not keyboard_ok:
        failures.append(('keyboard', width, scheme, after_end['view'], after_end['divider'], after_home['divider']))

    # Divider screenshot at 50% with the first listed move selected.
    evaluate(ws, "const s=document.getElementById('tp-divider'); s.value='50'; s.dispatchEvent(new Event('input'))")
    evaluate(ws, "document.querySelector('[data-macro]').value = "
                 "document.querySelector('[data-macro] option:nth-child(2)').value; "
                 "document.querySelector('[data-macro]').dispatchEvent(new Event('change'))")
    screenshot(ws, qa / f'{width}-{scheme}-divider.png')

    # Pointer: back in the displacement view, click the largest moved macro and expect it selected.
    evaluate(ws, "document.querySelector('[data-view=displacement]').click()")
    evaluate(ws, "document.querySelector('[data-macro]').value=''; "
                 "document.querySelector('[data-macro]').dispatchEvent(new Event('change'))")
    target = evaluate(ws, """(() => { const r = [...document.querySelectorAll('rect.tp-moved')].sort((a, b) =>
            b.getBBox().width * b.getBBox().height - a.getBBox().width * a.getBBox().height)[0];
        r.scrollIntoView({block: 'center'}); const b = r.getBoundingClientRect();
        return {x: b.left + b.width / 2, y: b.top + b.height / 2, index: r.dataset.index}; })()""")
    for kind in ('mouseMoved', 'mousePressed', 'mouseReleased'):
        ws.send('Input.dispatchMouseEvent', {'type': kind, 'x': target['x'], 'y': target['y'],
                                             'button': 'left', 'clickCount': 1})
    time.sleep(0.05)
    clicked = evaluate(ws, "document.querySelector('[data-macro]').value")
    if clicked != target['index']:
        failures.append(('pointer', width, scheme, target, clicked))

    errors = [e for e in ws.events if e.get('method') == 'Runtime.exceptionThrown'
              or (e.get('method') == 'Runtime.consoleAPICalled' and e['params']['type'] in ('error', 'warning'))
              or (e.get('method') == 'Log.entryAdded' and e['params']['entry']['level'] in ('error', 'warning'))]
    if errors:
        failures.append(('console', width, scheme, errors))
    print(f'{width}px {scheme}: body background {background}; button transition {transition}; '
          f'initial {initial["before"]!r} / {initial["after"]!r}; arrows {initial["arrows"]}; '
          f'keyboard ok={keyboard_ok}; pointer ok={clicked == target["index"]}; console messages {len(errors)}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--page', type=Path, default=INTERACTIVE / 'placement-comparison.html',
                        help='standalone explorer page to open')
    parser.add_argument('--data', type=Path, default=INTERACTIVE / 'placement-data.json',
                        help='explorer data; supplies the design names to visit')
    parser.add_argument('--qa-dir', type=Path, default=INTERACTIVE / 'qa',
                        help='where checks.json and the screenshots are written')
    parser.add_argument('--chrome', help='Chrome or Chromium executable (default: $CHROME or a standard location)')
    args = parser.parse_args()

    page, qa = args.page.resolve(), args.qa_dir.resolve()
    names = [record['benchmark'] for record in json.loads(args.data.read_text(encoding='utf-8'))]
    qa.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome(args.chrome)
    version = subprocess.run([chrome, '--version'], capture_output=True, text=True).stdout.strip()
    print(f'{version}; page {page}')

    profile = tempfile.mkdtemp(prefix='explorer-qa-chrome-')
    checks, failures = [], []
    proc = None
    try:
        proc, ws = start_chrome(chrome, profile)
        for width in WIDTHS:
            for scheme in SCHEMES:
                check_view(ws, page, qa, names, width, scheme, checks, failures)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)

    (qa / 'checks.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    expected = DESIGNS * len(WIDTHS) * len(SCHEMES)
    if len(checks) != expected:
        failures.append(('check count', len(checks), expected))
    print(f'{len(checks)} design/viewport/theme checks; {len(failures)} failures')
    for failure in failures:
        print('FAIL', json.dumps(failure, ensure_ascii=False)[:600])
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
