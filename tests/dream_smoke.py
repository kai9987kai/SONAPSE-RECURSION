"""Interactive dream validation. Uses only synthetic examples in a temporary browser profile."""

import json
import os
from pathlib import Path
import re
import shutil
import sys
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8794/"
artifacts = Path(sys.argv[2]) if len(sys.argv) > 2 else None
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE"))
    page = browser.new_page(viewport={"width": 1250, "height": 950})
    page.set_default_timeout(90000)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url.rstrip("/") + "/#dreaming", wait_until="domcontentloaded")
    # Instrument timers solely to control exactly when each synthetic tick occurs.
    page.evaluate("""()=>{const original=window.setInterval;window.setInterval=(fn,ms,...rest)=>{
      if(ms===240 && document.querySelector('#dreamWake').disabled===false)return original(()=>{},ms);
      return original(fn,ms,...rest)
    }}""")

    def frame():
        return int(
            re.search(r"Frame (\d+)", page.locator("#dreamReadout").inner_text())[1]
        )

    def start(seed, theme):
        page.fill("#dreamSeed", str(seed))
        page.select_option("#dreamTheme", theme)
        page.click("#dreamStart")
        page.wait_for_function(
            "document.querySelector('#dreamReadout').textContent.includes('Frame 1 ·')"
        )
        page.click("#dreamPause")
        return page.locator("#mixDream").evaluate("c=>c.toDataURL()")

    first = start(41, "Drift")
    assert "Entirely procedural" in page.locator("#dreamReadout").inner_text()
    page.click("#dreamStep")
    assert frame() == 2
    assert first != page.locator("#mixDream").evaluate("c=>c.toDataURL()")
    page.click("#dreamWake")
    assert page.locator("#dreamWake").is_disabled()
    assert first == start(41, "Drift")
    assert first != start(42, "Drift")
    page.click("#dreamWake")
    for source in ("julia", "orbit"):
        page.select_option("#mixExample", source)
        page.click("#mixAdd")
        page.wait_for_function(
            "document.querySelector('#studioArchive').textContent.includes('Next-frame / hold')"
        )
        page.click("#mixStop")
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#mixEpisode').options.length>=3")
    before = int(
        re.search(r"(\d+) studio", page.locator("#studioArchive").inner_text())[1]
    )
    page.fill("#dreamSeed", "6301")
    page.click("#dreamStart")
    page.wait_for_function(
        "document.querySelectorAll('#dreamGallery figure').length>=3"
    )
    page.click("#dreamPause")
    current = frame()
    page.wait_for_timeout(550)
    assert frame() == current
    assert (
        page.locator("#studioArchive")
        .inner_text()
        .startswith("Dreaming · 0 active inputs")
    )
    with page.expect_download() as event:
        page.click("#dreamJournal")
    journal = json.loads(Path(event.value.path()).read_text())
    assert (
        journal["seed"] == 6301
        and journal["synthetic"]
        and len(journal["entries"]) >= 3
    )
    for scene in journal["entries"]:
        assert scene["eligibleMemories"] == before
        assert {r["source"] for r in scene["fragments"]} == {"Julia set", "Orbit"}
    assert (
        int(re.search(r"(\d+) studio", page.locator("#studioArchive").inner_text())[1])
        == before
    )
    with page.expect_download() as event:
        page.click("#dreamStrip")
    image = Path(event.value.path())
    assert image.read_bytes()[:8] == bytes.fromhex("89504e470d0a1a0a")
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, artifacts / "example-strip.png")
        (artifacts / "example-journal.json").write_text(
            json.dumps(journal, indent=2), encoding="utf8"
        )
    page.click("#dreamWake")
    page.click("#mixImagine")
    page.wait_for_function(
        "document.querySelector('#studioArchive').textContent.startsWith('Imagining')"
    )
    page.click("#mixStop")
    assert not errors, errors
    print(
        f"Dream replay, new scenes without memory, two-episode recombination, pause, wake, exports and archive isolation passed ({before} observed frames).",
        flush=True,
    )
    browser.close()
