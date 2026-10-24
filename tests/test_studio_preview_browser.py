"""Opt in with HYDRAPY_BROWSER_TESTS=1; requires Playwright and Chromium."""
import os
import threading
import time

import pytest

from hydrapy.studio import Session, make_server

pytestmark = pytest.mark.skipif(os.environ.get('HYDRAPY_BROWSER_TESTS') != '1',
                                reason='set HYDRAPY_BROWSER_TESTS=1 for real Chromium checks')


@pytest.fixture
def workspace(tmp_path):
    from playwright.sync_api import sync_playwright
    (tmp_path / 'patch.py').write_text('from hydrapy import *\nsolid(.8,0,0).out(o0)\nsolid(0,.8,0).out(o1)\n')
    session = Session(tmp_path, 'patch.py', backend='cpu', size=(64, 36), fps=30)
    server, token = make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            yield session, page, f'http://127.0.0.1:{server.server_port}/#{token}'
            assert not errors
            browser.close()
    finally:
        server.shutdown(); server.server_close(); session.close()


def test_browser_rate_switch_pause_hide_and_reconnect(workspace):
    session, page, url = workspace
    page.goto(url)
    page.wait_for_function('window.hydrapyPreview.frames >= 5')
    start = page.evaluate('hydrapyPreview.frames')
    page.wait_for_timeout(1200)
    assert page.evaluate('hydrapyPreview.frames') - start >= 10  # clears the former 4 fps cap
    page.get_by_role('button', name='Show output 1', exact=True).click()
    page.wait_for_function('hydrapyPreview.samples.at(-1).output === "o1"')
    assert page.evaluate('''() => {
      const c=document.createElement('canvas'); c.width=64; c.height=36;
      const ctx=c.getContext('2d'); ctx.drawImage(document.getElementById('preview'),0,0);
      const p=ctx.getImageData(0,0,1,1).data; return p[1]>180 && p[0]<10;
    }''')
    page.locator('#play').click()
    page.wait_for_timeout(200)
    held = page.evaluate('hydrapyPreview.frames')
    page.wait_for_timeout(1200)
    assert page.evaluate('hydrapyPreview.frames') == held
    page.locator('#play').click()
    page.wait_for_function('(n) => hydrapyPreview.frames > n', arg=held)
    # Exercise the visibility handler without depending on headless OS window focus.
    page.evaluate('''window.testHidden=true;
      Object.defineProperty(document,'hidden',{get:()=>window.testHidden});
      document.dispatchEvent(new Event('visibilitychange'));''')
    hidden = page.evaluate('hydrapyPreview.frames')
    rendered = session.renderer.frame
    page.wait_for_timeout(250)
    assert page.evaluate('hydrapyPreview.frames') == hidden
    assert session.renderer.frame > rendered
    page.evaluate('testHidden=false; document.dispatchEvent(new Event("visibilitychange"))')
    page.wait_for_function('(n) => hydrapyPreview.frames > n', arg=hidden)
    page.reload()
    page.wait_for_function('hydrapyPreview.frames >= 3')
    stats = page.evaluate('hydrapyPreview')
    assert not stats['error']
    assert all(row['decode_ms'] >= 0 and row['transfer_ms'] >= 0 for row in stats['samples'])


def test_delayed_decode_and_output_switch_never_queue_requests(workspace):
    session, page, url = workspace
    page.add_init_script('''const original=HTMLImageElement.prototype.decode;
      HTMLImageElement.prototype.decode=async function(){
        await original.call(this); await new Promise(resolve=>setTimeout(resolve,180));
      };''')
    active, peak = set(), []
    def request(r):
        if '/api/frame?' in r.url:
            active.add(r); peak.append(len(active))
    page.on('request', request)
    page.on('requestfinished', lambda r: active.discard(r))
    page.on('requestfailed', lambda r: active.discard(r))
    page.goto(url)
    page.wait_for_function('hydrapyPreview.frames >= 2')
    start = session.renderer.frame
    page.get_by_role('button', name='Show output 1', exact=True).click()
    page.wait_for_function('hydrapyPreview.samples.at(-1).output === "o1"')
    n = page.evaluate('hydrapyPreview.samples.length')
    page.wait_for_timeout(750)
    rows = page.evaluate('hydrapyPreview.samples')
    assert all(row['output'] == 'o1' for row in rows[n:])
    assert max(peak) == 1
    assert session.renderer.frame - start > len(rows[n:]) * 2
    assert session.preview.snapshot()['pending'] <= 1
    assert session.preview.snapshot()['thumbnail_pending'] <= 4
