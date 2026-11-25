"""Real-browser validation of concurrent inputs, archive recall, and Python replay."""

import json
import os
import re
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
    for name in ("julia", "mandelbrot", "carpet"):
        page.select_option("#mixExample", name)
        page.click("#mixAdd")
    page.set_input_files(
        "#mixFiles",
        [root / "web/stimuli/ribbons.webm", root / "web/stimuli/interference.webm"],
    )
    page.route(
        "**/studio-test-stream.webm",
        lambda route: route.fulfill(
            path=root / "web/stimuli/ripple.webm", content_type="video/webm"
        ),
    )
    page.fill("#mixUrl", url.rstrip("/") + "/studio-test-stream.webm")
    page.click("#mixStream")
    page.click("#mixCamera")
    page.wait_for_function(
        "document.querySelectorAll('#studioSources button').length===7"
    )
    page.select_option("#mixResolution", "640")
    page.wait_for_function(
        "document.querySelector('#studioArchive').textContent.includes('Next-frame / hold')"
    )
    assert page.locator("#mixDream").evaluate("c=>c.width===640&&c.height===384")
    assert page.locator("#mixDream").evaluate(
        "c=>c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4!==3&&v>0)"
    )
    page.get_by_role("button", name="Remove Webcam").click()
    assert page.locator("#studioSources button").count() == 6
    page.click("#mixStop")
    with page.expect_download() as export:
        page.click("#mixExport")
    file = export.value.path()
    frames = json.loads(Path(file).read_text())["frames"]
    assert any(
        "Webcam" in f["source"]
        and "carpet" in f["source"].lower()
        and "ribbons.webm" in f["source"]
        and f["width"] == 640
        for f in frames
    )
    run = subprocess.run(
        [sys.executable, str(root / "scripts/visual_replay.py"), str(file)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (run.stdout, run.stderr)
    print(
        "Seven concurrent inputs, 640 x 384 prediction, source removal and Python replay:",
        run.stdout,
        flush=True,
    )
    page.reload()
    page.select_option("#mixResolution", "1920")
    page.select_option("#mixHorizon", "4")
    page.click("#mixImagine")
    page.wait_for_function(
        "document.querySelector('#studioArchive').textContent.startsWith('Imagining') && !document.querySelector('#studioArchive').textContent.includes('· 0 memories used')"
    )
    with page.expect_download() as png:
        page.click("#mixImage")
    header = Path(png.value.path()).read_bytes()[:24]
    assert header[:8] == bytes.fromhex("89504e470d0a1a0a")
    assert (
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:24], "big"),
    ) == (1920, 1080)
    summary = page.locator("#studioArchive").inner_text()
    assert int(re.search(r"(\d+) studio", summary)[1]) == len(frames)
    assert int(re.search(r"(\d+) memories used", summary)[1]) == len(frames)
    page.click("#mixStop")
    # Populate older entries through the real v1 archive schema to exercise a long history quickly.
    page.evaluate("""async () => {
      const db=await new Promise(ok=>{const r=indexedDB.open('sonapse-visual-studio-v1');r.onsuccess=()=>ok(r.result)});
      const req=s=>new Promise(ok=>{s.onsuccess=()=>ok(s.result)});
      const rows=await req(db.transaction('index').objectStore('index').getAll());
      const blob=await req(db.transaction('images').objectStore('images').get(rows[0].id));
      const tx=db.transaction(['index','images'],'readwrite');
      for(let i=0;i<251;i++){
        const f={...rows[0],epoch:'history',tick:i,id:'history:'+i,source:'History fixture'};
        tx.objectStore('index').put(f);tx.objectStore('images').put(blob,f.id);
      }
      await new Promise(ok=>tx.oncomplete=ok);db.close();
      const classic={epoch:"old",tick:0,source:"Classic fixture",pixels:Array(720).fill(127),state:Array(18).fill(0),host:[0,0,0]};
      localStorage.setItem("sonapse.visual.episodes.v2",JSON.stringify([classic]));
    }""")
    page.reload()
    page.click("#mixImagine")
    expected = len(frames) + 251 + 1
    page.wait_for_function(
        "n=>document.querySelector('#studioArchive').textContent.includes(n+' memories used')",
        arg=expected,
    )
    assert not errors, errors
    print(
        "Reloaded archive and autonomous all-memory context passed:",
        expected,
        "observations.",
        flush=True,
    )
    browser.close()
