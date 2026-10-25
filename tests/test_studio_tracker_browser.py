"""Opt-in real Chromium checks for Gorgon's integrated control surface."""
import json
import os
import threading
import time
from pathlib import Path

import pytest

from hydrapy.studio import Session, make_server

pytestmark = pytest.mark.skipif(os.environ.get('HYDRAPY_BROWSER_TESTS') != '1',
                                reason='set HYDRAPY_BROWSER_TESTS=1 for real Chromium checks')


@pytest.fixture
def workspace(tmp_path):
    from playwright.sync_api import sync_playwright
    (tmp_path/'p.py').write_text("from hydrapy import *\nz=Param('Dolly Z', .35)\nr=Param('Roll',0,min=-45,max=45)\nf=Param('Feedback',.25)\nsolid(z,0,0).out(o0)\nsolid(0,.8,0).out(o1)\n")
    session = Session(tmp_path, 'p.py', backend='cpu', size=(64,36), fps=30)
    server, token = make_server(session, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={'width':1440, 'height':1000})
            errors = []
            context.on('page', lambda page: page.on('pageerror', lambda error: errors.append(str(error))))
            page = context.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/tracker/#{token}')
            page.wait_for_function('window.gorgonPreview?.frames >= 3')
            yield session, page, context, tmp_path
            assert not errors
            browser.close()
    finally:
        server.shutdown(); server.server_close(); session.close()


def test_score_edit_seek_tempo_save_reload_and_real_patch(workspace):
    session, page, context, root = workspace
    page.get_by_role('button', name='Play', exact=True).click()
    page.get_by_role('button', name='Pause', exact=True).wait_for()
    page.get_by_role('button', name='Pause', exact=True).click()
    page.get_by_role('button', name='Play', exact=True).wait_for()
    clock = session.clock_time
    page.get_by_role('button', name='Select pattern 3: ASCENT', exact=True).click()
    page.wait_for_function("document.getElementById('position').textContent.startsWith('009')")
    assert session.clock_time == clock  # tracker seek did not seek the renderer
    page.locator('#tempo').fill('90')
    page.locator('#tempo').press('Tab')
    page.wait_for_timeout(350)
    assert session.tracker.document['bpm'] == 90
    page.locator('#zSlider').fill('0.8')
    page.wait_for_timeout(350)
    assert session.tracker.document['patterns'][2]['keys']['z'][0]['v'] == .8
    page.get_by_role('button', name='Play', exact=True).click()
    page.wait_for_timeout(120)
    page.get_by_role('button', name='Pause', exact=True).click()
    assert session.renderer.patch.controls['Dolly Z'].base > .79
    with page.expect_download() as download:
        page.locator('#saveBtn').click()
    data = json.loads(open(download.value.path()).read())
    assert data == json.loads((root/'.hydrapy/tracker.json').read_text())
    page.reload()
    page.wait_for_function('gorgonPreview.frames>=1')  # paused sessions deliver the held frame once
    assert page.locator('#tempo').input_value() == '90'
    assert page.locator('#bind_z').input_value() == 'Dolly Z'
    page.screenshot(path='/tmp/gorgon-integrated-desktop.png')


def test_independent_output_windows_routing_blackout_and_closed_controller(workspace):
    session, page, context, root = workspace
    with page.expect_popup() as window:
        page.locator('#auxBtn').click()
    aux = window.value
    aux.locator('#outputDismiss').click()
    aux.wait_for_function('gorgonPreview.frames>=3 && gorgonPreview.output==="o1"')
    assert session.renderer.patch.display.name == 'o0'
    assert aux.evaluate("document.getElementById('outputCanvas').getContext('2d').getImageData(600,400,1,1).data[1]") > 195
    with page.expect_popup() as window:
        page.locator('#programBtn').click()
    program = window.value
    program.locator('#outputDismiss').click()
    program.wait_for_function('gorgonPreview.frames>=3 && gorgonPreview.output==="o0"')
    page.locator('#auxMode').select_option('o0')
    aux.wait_for_function('gorgonPreview.output==="o0"')
    page.locator('#blackoutBtn').click()
    aux.wait_for_function("document.getElementById('outputCanvas').getContext('2d').getImageData(600,400,1,1).data[0]===0")
    assert not session.paused
    page.locator('#blackoutBtn').click()
    aux.wait_for_function("document.getElementById('outputCanvas').getContext('2d').getImageData(600,400,1,1).data[0]>60")
    page.locator('#blackoutBtn').focus()
    page.keyboard.press('b');page.keyboard.press('b')
    page.wait_for_timeout(400)
    assert not session.tracker_blackout  # rapid toggles use acknowledged state
    page.close()
    before = aux.evaluate('gorgonPreview.frames')
    aux.wait_for_function('(n)=>gorgonPreview.frames>n+3', arg=before)
    assert not session.paused


def test_import_validation_hold_boundary_and_slow_decode_are_bounded(workspace):
    session, page, context, root = workspace
    d = json.loads(json.dumps(session.tracker.document))
    d['patterns'][0]['easing']['z']='hold'
    d['patterns'][0]['keys']['z']=[{'r':0,'v':.1},{'r':16,'v':.9}]
    file = root/'score.json'; file.write_text(json.dumps(d))
    page.locator('#importFile').set_input_files(file)
    page.wait_for_function("document.getElementById('toast').textContent.startsWith('Score loaded')")
    page.locator('[data-row="16"][data-col="4"]').click()
    assert page.locator('#zValue').text_content() == '0.90'
    accepted = session.tracker.document
    d['bpm'] = 900;file.write_text(json.dumps(d))
    page.locator('#importFile').set_input_files(file)
    page.wait_for_function("document.getElementById('toast').textContent.includes('Tempo')")
    assert session.tracker.document is accepted
    page.add_init_script('''const decode=HTMLImageElement.prototype.decode;
      HTMLImageElement.prototype.decode=async function(){await decode.call(this);await new Promise(r=>setTimeout(r,180));};''')
    active, peak = set(), []
    def request(r):
        if '/api/frame?' in r.url: active.add(r);peak.append(len(active))
    page.on('request',request)
    page.on('requestfinished',lambda r:active.discard(r))
    page.on('requestfailed',lambda r:active.discard(r))
    page.reload();page.wait_for_function('gorgonPreview.frames>=3')
    assert max(peak)==1
    assert session.preview.snapshot()['pending']<=1
    assert len(page.evaluate('gorgonPreview.samples'))<=300


def test_pitched_note_edit_audition_frequency_and_reload(workspace):
    session, page, context, root = workspace
    # Select an existing event without erasing it, then enter a note beyond the
    # original demo's small fixed vocabulary.
    page.locator('[data-row="6"][data-col="1"]').click()
    assert session.tracker.document['patterns'][0]['audio'][1][6] is not None
    page.locator('#noteInput').fill('E4'); page.locator('#noteInput').press('Tab')
    page.locator('#velocityInput').fill('110'); page.locator('#velocityInput').press('Tab')
    page.wait_for_timeout(400)
    assert session.tracker.document['patterns'][0]['audio'][1][6] == {'note': 'E-4', 'vel': 110}
    values = page.evaluate('''async()=>{const n=await import('/tracker/notes.js');
      return [n.noteMidi('E4'), n.noteFrequency('A-4',2), n.noteFrequency('A-5',2), n.noteFrequency('A-4',7)];}''')
    assert values[:3] == [64, 440, 880]
    assert values[3] == pytest.approx(440*2**(5/12))
    page.evaluate('''() => {
      window.testOscillators=[];
      const make=AudioContext.prototype.createOscillator;
      AudioContext.prototype.createOscillator=function(){const o=make.call(this);testOscillators.push(o);return o;};
    }''')
    page.locator('#auditionBtn').click()
    page.wait_for_function('testOscillators.length===2')
    frequency = 440 * 2 ** ((64 - 69) / 12)
    assert page.evaluate('testOscillators.map(o=>o.frequency.value)') == pytest.approx([frequency, 2 * frequency])
    with page.expect_download() as download:
        page.locator('#saveBtn').click()
    saved = json.loads(Path(download.value.path()).read_text())
    assert saved['patterns'][0]['audio'][1][6]['note'] == 'E-4'
    assert page.locator('.inspector').evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
    page.screenshot(path='/tmp/gorgon-pitched-voices.png')
    page.reload(); page.wait_for_function('gorgonPreview.frames>=1')
    page.locator('[data-row="6"][data-col="1"]').click()
    assert page.locator('#noteInput').input_value() == 'E-4'
    assert page.locator('#velocityInput').input_value() == '110'


def test_download_uses_exact_saved_score_after_studio_output_sync(workspace):
    session, page, context, root = workspace
    # Freeze the editor's GET snapshot while Studio changes its output. Saving
    # an otherwise unedited older revision is allowed by the sync protocol.
    snapshot = dict(session.tracker_state, server_now=time.monotonic())
    page.route('**/api/tracker', lambda route: route.fulfill(json=snapshot))
    session.call('output', index=1)
    with page.expect_download() as download:
        page.locator('#saveBtn').click()
    saved = json.loads(Path(download.value.path()).read_text())
    assert saved == json.loads((root/'.hydrapy/tracker.json').read_text())
    assert saved['modes']['program'] == 'o1'
