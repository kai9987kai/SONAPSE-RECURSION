"""Run against the local sensory dashboard with Chromium's synthetic capture device."""

import os
import json
import subprocess
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8794/"
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
        args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
    )
    page = browser.new_page(permissions=["camera"])
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    for name in ("orbit", "ripple", "crossing"):
        page.select_option("#senseExample", name)
        page.click("#sensePlay")
        page.wait_for_function(
            "document.querySelector('#senseLog').textContent.includes('Observing '+document.querySelector('#senseExample').value) && document.querySelector('#senseLog').textContent.includes('Forecast / hold MAE')"
        )
    page.set_input_files("#senseFile", str(root / "web/stimuli/ripple.webm"))
    page.wait_for_function(
        "document.querySelector('#senseLog').textContent.includes('Observing ripple.webm')"
    )
    page.click("#senseCam")
    page.wait_for_function(
        "document.querySelector('#senseLog').textContent.includes('Observing Webcam')"
    )
    stream = page.evaluate_handle("document.querySelector('#senseVideo').srcObject")
    page.click("#senseStop")
    assert stream.evaluate("s=>s.getTracks().every(t=>t.readyState==='ended')")
    page.click("#storeMemory")
    size = page.evaluate(
        "JSON.parse(localStorage.getItem('sonapse.visual.episodes.v2')).length"
    )
    assert size > 5
    with page.expect_download() as downloaded:
        page.click("#exportVisual")
    run = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/visual_replay.py"),
            str(downloaded.value.path()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print("Python replay", json.loads(run.stdout), flush=True)
    page.reload()
    assert page.locator("#recallEpisode option").count() >= 4
    page.locator("#recallEpisode").select_option(index=1)
    page.click("#freeDream")
    page.wait_for_function(
        "document.querySelector('#senseLog').textContent.includes('Imagining')"
    )
    assert page.evaluate(
        "document.querySelector('#dream').getContext('2d').getImageData(0,0,20,12).data.some((v,i)=>i%4!==3 && v>0)"
    )
    page.click("#storeMemory")
    assert (
        page.evaluate(
            "JSON.parse(localStorage.getItem('sonapse.visual.episodes.v2')).length"
        )
        == size
    )
    page.on("dialog", lambda d: d.accept())
    page.click("#clearMemory")
    assert page.evaluate("localStorage.getItem('sonapse.visual.episodes.v2')===null")
    assert not errors, errors
    print(
        "Passed three example videos, upload, synthetic webcam, capture release, reload, autonomous dream, memory isolation and clear.",
        flush=True,
    )
    browser.close()
