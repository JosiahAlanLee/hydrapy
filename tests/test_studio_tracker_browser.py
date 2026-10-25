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
    assert session.tracker.beat(session.clock_time) < 1  # selecting is no longer a seek
    page.locator('#seekCellBtn').click()
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


def test_output_polarity_and_audio_controls_are_saved_and_show_failures(workspace):
    from hydrapy.es40 import ES40CV
    from test_studio_audio import FakeClient
    session, page, context, root = workspace
    session.cv = ES40CV([dict(name='Digital pair', channels=[16, 17], header=1,
                             expander='8gt', map={'1': 'gate1'})], open_stream=False)
    session.audio.client_factory = FakeClient
    # This test exercises controls, requests and saved settings; JACK callbacks
    # and native server behavior have their own processing/integration tests.
    session.audio._produce = lambda: None
    select = page.get_by_label('ES-40 polarity', exact=False)
    select.wait_for()
    select.select_option('invert_left')
    page.wait_for_function("document.getElementById('polarityStatus').textContent.includes('inversion active')")
    saved = json.loads((root/'.hydrapy/outputs.json').read_text())
    assert list(saved['polarity'].values()) == ['invert_left']
    page.reload()
    page.wait_for_function('gorgonPreview.frames>=1')
    assert page.get_by_label('ES-40 polarity', exact=False).input_value() == 'invert_left'
    page.locator('#jackEnableBtn').click()
    page.get_by_role('button', name='Disable JACK audio ports', exact=True).wait_for()
    assert 'HydraPy:optical_L' in page.locator('#jackPorts').text_content()
    page.locator('#audioLead').select_option('4')
    page.wait_for_function("document.getElementById('jackStatus').textContent.includes('10.7 ms')")
    assert json.loads((root/'.hydrapy/outputs.json').read_text())['audio']['lead_blocks'] == 4
    page.locator('#jackEnableBtn').click()
    page.get_by_role('button', name='Enable JACK audio ports', exact=True).wait_for()
    def fail(*a, **kw): raise RuntimeError('test: no JACK server')
    session.audio.client_factory = fail
    page.locator('#jackEnableBtn').click()
    page.get_by_role('button', name='Retry JACK audio', exact=True).wait_for()
    assert 'no JACK server' in page.locator('#jackStatus').text_content()
    assert not json.loads((root/'.hydrapy/outputs.json').read_text())['audio']['enabled']


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


def test_held_previews_skip_repaints_but_resize_route_and_blackout_still_draw(workspace):
    session, page, context, root = workspace
    session.call('pause', paused=True)
    page.wait_for_timeout(300)  # let the last decoded frame reach both canvases
    assert not session.preview.thumbnails  # Gorgon does not use Studio's output picker
    page.evaluate('''() => {
      window.previewPaints={};
      const draw=CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage=function(...args){
        previewPaints[this.canvas.id]=(previewPaints[this.canvas.id]||0)+1;
        return draw.apply(this,args);
      };
    }''')
    page.wait_for_timeout(250)
    assert page.evaluate('previewPaints.previewCanvas||0') == 0
    assert page.evaluate('previewPaints.sceneThumb||0') == 0

    page.set_viewport_size({'width':1200, 'height':900})
    page.wait_for_function('(previewPaints.previewCanvas||0)>0')
    pixel = """() => {const c=document.getElementById('previewCanvas');
      return Array.from(c.getContext('2d').getImageData(c.width/2,c.height/2,1,1).data);} """
    assert page.evaluate(pixel)[0] > 60
    session.call('output', index=1)
    page.wait_for_function('gorgonPreview.output === "o1"')
    page.wait_for_function('(' + pixel + ')()[1] > 195')
    page.locator('#blackoutBtn').click()
    page.wait_for_function('(' + pixel + ')().slice(0,3).every(v=>v===0)')
    page.locator('#blackoutBtn').click()
    page.wait_for_function('(' + pixel + ')()[1] > 195')


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


def test_browser_synth_continues_across_repeated_pattern_wraps(workspace):
    session, page, context, root = workspace
    document = json.loads(json.dumps(session.tracker.document))
    document.update(bpm=200, loop=True, pattern=0)
    for pattern in document['patterns']:
        pattern['audio'] = [[None] * 64 for _ in range(4)]
    document['patterns'][0]['audio'][1][0] = {'note': 'A-4', 'vel': 100}
    file = root/'loop.json'; file.write_text(json.dumps(document))
    page.locator('#importFile').set_input_files(file)
    page.wait_for_function("document.getElementById('toast').textContent.startsWith('Score loaded')")
    page.evaluate('''() => {
      window.synthOscillators=0;
      const create=AudioContext.prototype.createOscillator;
      AudioContext.prototype.createOscillator=function(){synthOscillators++;return create.call(this);};
    }''')
    page.locator('#audioBtn').click()
    page.get_by_role('button', name='Play', exact=True).click()
    # Two oscillators per FM note, once per 4.8-second loop.
    page.wait_for_function('synthOscillators>=6', timeout=17000)
    assert page.evaluate('synthOscillators') == 6


def test_keyboard_entry_focus_and_edit_cursor_do_not_seek(workspace):
    session, page, context, root = workspace
    session.call('tracker_transport', enabled=True, playing=False, beat=8)
    page.wait_for_timeout(250)
    transport = session.tracker.transport_revision
    page.locator('[data-row="1"][data-col="1"]').click()
    page.keyboard.press('Enter'); page.keyboard.press('Enter')
    assert page.evaluate('document.activeElement.dataset.col') == '1'
    page.wait_for_timeout(250)
    assert session.tracker.document['patterns'][0]['audio'][1][1] is None
    page.locator('#entryStep').select_option('4')
    page.locator('#pianoBtn').click()
    page.keyboard.press('z'); page.keyboard.press('s')
    assert page.evaluate('document.activeElement.dataset.row') == '9'
    page.wait_for_timeout(250)
    notes = session.tracker.document['patterns'][0]['audio'][1]
    assert notes[1]['note'] == 'C-4' and notes[5]['note'] == 'C#4'
    page.keyboard.press('b')  # piano entry takes precedence over Blackout inside the grid
    page.wait_for_timeout(250)
    assert not session.tracker_blackout
    assert session.tracker.document['patterns'][0]['audio'][1][9]['note'] == 'G-4'
    page.locator('[data-row="2"][data-col="4"]').click()
    page.locator('#cellValue').fill('0.82'); page.locator('#cellValue').press('Tab')
    page.get_by_role('button', name='Select pattern 3: ASCENT', exact=True).click()
    page.wait_for_timeout(250)
    assert session.tracker.transport_revision == transport
    assert session.tracker.beat(session.clock_time) == 8
    assert session.tracker.document['patterns'][0]['keys']['z'][1] == {'r': 2, 'v': .82}
    page.locator('#seekCellBtn').click()
    page.wait_for_function("document.getElementById('position').textContent.startsWith('009')")
    assert session.tracker.transport_revision > transport


def test_mixed_block_edit_transpose_undo_redo_and_transport_preservation(workspace):
    session, page, context, root = workspace
    transport = session.tracker.transport_revision
    for row, note in [(1, 'C4'), (2, 'D4')]:
        page.locator(f'[data-row="{row}"][data-col="1"]').click()
        page.locator('#noteInput').fill(note); page.locator('#noteInput').press('Tab')
    page.locator('[data-row="1"][data-col="1"]').click()
    page.locator('[data-row="2"][data-col="2"]').click(modifiers=['Shift'])
    assert page.locator('#selectionInfo').inner_text() == '2 × 2 cells'
    page.locator('#copyBtn').click()
    page.locator('[data-row="8"][data-col="1"]').click(); page.locator('#pasteBtn').click()
    page.locator('[data-row="8"][data-col="1"]').click()
    page.locator('[data-row="9"][data-col="2"]').click(modifiers=['Shift'])
    page.locator('#velocityInput').fill('77');page.locator('#velocityInput').press('Tab')
    page.locator('#applyVelocityBtn').click()
    page.locator('[data-transpose="12"]').click()
    page.locator('#undoBtn').click(); page.locator('#redoBtn').click()
    page.wait_for_timeout(350)
    notes = session.tracker.document['patterns'][0]['audio'][1]
    assert notes[1]['note'] == 'C-4' and notes[2]['note'] == 'D-4'
    assert notes[8]['note'] == 'C-5' and notes[9]['note'] == 'D-5'
    assert notes[8]['vel'] == notes[9]['vel'] == 77
    assert session.tracker.document['patterns'][0]['audio'][2][9] is None
    assert session.tracker.transport_revision == transport
    assert page.evaluate('document.activeElement.dataset.col') == '2'
    page.locator('[data-row="9"][data-col="1"]').click()
    page.keyboard.press('Shift+ArrowDown')
    assert page.locator('#selectionInfo').inner_text() == '2 × 1 cells'
    page.keyboard.press('Delete'); page.keyboard.press('Control+z')
    page.wait_for_timeout(250)
    assert session.tracker.document['patterns'][0]['audio'][1][9]['note'] == 'D-5'


def test_pattern_tools_import_undo_and_explicit_launch(workspace):
    session, page, context, root = workspace
    page.locator('#patternMenu summary').click()
    page.locator('#patternName').fill('Study <1>'); page.locator('#patternName').press('Tab')
    page.locator('#duplicateTarget').select_option('1')
    page.locator('#duplicatePatternBtn').click()
    page.wait_for_timeout(250)
    assert session.tracker.document['patterns'][0] == session.tracker.document['patterns'][1]
    page.get_by_role('button', name='Select pattern 2: Study <1>', exact=True).click()
    page.locator('#patternMenu summary').click(); page.locator('#clearPatternBtn').click()
    assert page.locator('[data-row="0"][data-col="0"]').inner_text() == '···'
    page.locator('#undoBtn').click()
    page.wait_for_timeout(250)
    assert session.tracker.document['patterns'][1]['audio'][0][0] is not None
    document = json.loads(json.dumps(session.tracker.document)); document['bpm'] = 99
    file = root / 'other.json'; file.write_text(json.dumps(document))
    page.locator('#importFile').set_input_files(file)
    page.wait_for_function("document.getElementById('tempo').value==='99'")
    page.locator('#undoBtn').click()
    page.wait_for_function("document.getElementById('tempo').value==='138'")
    page.get_by_role('button', name='Play from pattern 2', exact=True).click()
    page.get_by_role('button', name='Pause', exact=True).wait_for()
    assert session.tracker.enabled and not session.paused
    assert session.tracker.beat(session.clock_time) >= 16


def test_editor_and_file_actions_remain_accessible_at_laptop_and_phone_sizes(workspace):
    session, page, context, root = workspace
    page.set_viewport_size({'width': 1366, 'height': 768})
    page.locator('[data-row="1"][data-col="1"]').click()
    for selector in ['#noteInput', '#velocityInput', '#seekCellBtn', '#saveBtn']:
        box = page.locator(selector).bounding_box()
        assert box and 0 <= box['y'] and box['y'] + box['height'] < 768, selector
    page.screenshot(path='/tmp/gorgon-editor-laptop.png')
    for width in [960, 390]:
        page.set_viewport_size({'width': width, 'height': 844})
        assert page.locator('#saveBtn').is_visible()
        page.locator('#fileMenu summary').click()
        for selector in ['#importBtn', '#restoreBtn', '#studioLink', '#choosePatchBtn', '#downloadScoreBtn']:
            assert page.locator(selector).is_visible(), (width, selector)
        page.locator('#fileMenu summary').click()
    page.locator('#viewMenu summary').click(); page.locator('#textSize').select_option('large')
    assert page.locator('.cell').first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)') >= 14
    page.locator('#viewMenu summary').click()
    page.locator('[data-row="1"][data-col="1"]').click()
    page.keyboard.press('ArrowRight');page.keyboard.press('ArrowRight');page.keyboard.press('ArrowRight');page.keyboard.press('ArrowRight')
    assert page.evaluate('document.activeElement.dataset.col') == '5'
    page.screenshot(path='/tmp/gorgon-editor-phone.png', full_page=True)


@pytest.mark.parametrize('accepted_before_disconnect', [False, True])
def test_edit_recovery_after_transport_failure(workspace, accepted_before_disconnect):
    session, page, context, root = workspace
    def interrupt(route):
        if accepted_before_disconnect:
            route.fetch()  # server accepted the edit, but the reply never reached the controller
        route.fulfill(status=503, json={'error': 'Simulated connection interruption'})
    page.route('**/api/tracker_score', interrupt, times=1)
    page.locator('[data-row="1"][data-col="1"]').click()
    page.locator('#noteInput').fill('E4');page.locator('#noteInput').press('Tab')
    page.locator('#editConnection').wait_for(state='visible')
    with page.expect_download() as download:
        page.locator('#recoveryBtn').click()
    recovered = json.loads(Path(download.value.path()).read_text())
    assert recovered['patterns'][0]['audio'][1][1]['note'] == 'E-4'
    page.locator('#retryScoreBtn').click()
    page.locator('#editConnection').wait_for(state='hidden')
    page.wait_for_timeout(250)
    assert session.tracker.document['patterns'][0]['audio'][1][1]['note'] == 'E-4'
    with page.expect_download() as download:
        page.locator('#saveBtn').click()
    assert json.loads(Path(download.value.path()).read_text()) == recovered


def test_revision_conflict_keeps_local_recovery_without_overwriting_other_controller(workspace):
    session, page, context, root = workspace
    snapshot = dict(session.tracker_state, server_now=time.monotonic())
    page.route('**/api/tracker', lambda route: route.fulfill(json=snapshot))
    other = json.loads(json.dumps(session.tracker.document));other['bpm'] = 110
    session.call('tracker_score', document=other, revision=session.tracker.revision)
    page.locator('[data-row="1"][data-col="1"]').click()
    page.locator('#noteInput').fill('F4');page.locator('#noteInput').press('Tab')
    page.locator('#editConnection').wait_for(state='visible')
    assert not page.locator('#retryScoreBtn').is_visible()
    with page.expect_download() as download:
        page.locator('#saveBtn').click()  # Save falls back to a local recovery file
    recovered = json.loads(Path(download.value.path()).read_text())
    assert recovered['patterns'][0]['audio'][1][1]['note'] == 'F-4'
    assert session.tracker.document['bpm'] == 110
    assert session.tracker.document['patterns'][0]['audio'][1][1] is None
    # Freeze the score only until its conflicting edit is rejected; subsequent
    # transport checks must see live acknowledgements rather than a frozen poll.
    page.unroute('**/api/tracker')
    page.locator('#blackoutBtn').click()
    page.wait_for_function("document.getElementById('blackoutBtn').getAttribute('aria-pressed')==='true'")
    assert session.tracker_blackout  # independent of the rejected edit
    page.locator('#blackoutBtn').click()
    page.once('dialog', lambda dialog: dialog.accept())
    page.locator('#reloadScoreBtn').click()
    page.locator('#editConnection').wait_for(state='hidden')
    assert page.locator('#tempo').input_value() == '110'


SCENE_PATCH = ("from hydrapy import *\nz=Param('Dolly Z', .35)\nr=Param('Roll',0,min=-45,max=45)\nf=Param('Feedback',.25)\n"
               "@scene('drift')\ndef _(): return osc(8).rotate(r)\n"
               "@scene('fracture')\ndef _(): return voronoi(5).kaleid(6)\n"
               "@scene('glow', o1)\ndef _(): return noise(3)\n")


def test_scene_column_entry_fade_copy_undo_playback_and_missing_names(workspace):
    session, page, context, root = workspace
    (root/'s.py').write_text(SCENE_PATCH)
    session.call('load', path='s.py')
    page.wait_for_function("document.querySelector('[data-row=\"0\"][data-col=\"7\"]') && "
                           "document.querySelector('.track-header.scene')")
    cell = lambda r: page.locator(f'[data-row="{r}"][data-col="7"]')
    scenes = lambda: session.tracker.document['patterns'][0]['scenes']
    # Enter steps through the patch's scenes: drift, then fracture after it.
    cell(0).click(); page.wait_for_function("document.getElementById('sceneChoice').options.length === 3")
    page.keyboard.press('Enter')
    cell(16).click(); page.keyboard.press('Enter')
    page.locator('#sceneFade').fill('8'); page.locator('#sceneFade').press('Tab')
    page.wait_for_timeout(350)
    assert scenes() == [{'r': 0, 'name': 'drift', 'fade': 0}, {'r': 16, 'name': 'fracture', 'fade': 8}]
    assert 'F08' in cell(16).inner_text()
    assert page.locator('#sceneChoice option').all_inner_texts() == ['drift · o0', 'fracture · o0', 'glow · o1']
    # A scene block copies only into the SCENE column; undo takes it back.
    cell(16).click(); page.locator('#copyBtn').click()
    cell(32).click(); page.locator('#pasteBtn').click()
    page.locator(f'[data-row="40"][data-col="6"]').click(); page.locator('#pasteBtn').click()
    page.wait_for_timeout(350)
    assert [k['r'] for k in scenes()] == [0, 16, 32]
    page.locator('#undoBtn').click(); page.wait_for_timeout(350)
    assert [k['r'] for k in scenes()] == [0, 16]
    # Playing drives the patch's scene weights: fracture fades in over 2 beats from beat 4.
    session.call('tracker_transport', enabled=True, playing=True, beat=4.5)
    page.wait_for_timeout(150)
    session.call('tracker_transport', playing=False)
    weights = dict(session.renderer.patch.scene_weights)
    assert weights['drift'] == 1 and 0 < weights['fracture'] < 1 and weights['glow'] == 1
    # A name the loaded patch doesn't declare is struck through, and kept.
    (root/'s.py').write_text(SCENE_PATCH.replace("'fracture'", "'shards'"))
    session.call('load', path='s.py')
    page.wait_for_function("document.querySelector('[data-row=\"16\"][data-col=\"7\"]').classList.contains('missing')")
    assert scenes()[1]['name'] == 'fracture'
    # Arrow keys reach the SCENE column and stop there.
    cell(16).click(); page.keyboard.press('ArrowRight')
    assert page.evaluate('document.activeElement.dataset.col') == '7'
