"""Optional standalone browser integration test. Requires Playwright and Chromium."""

import os
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE"))
    page = browser.new_page(viewport={"width": 980, "height": 800}, offline=True)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto((root / "web/index.html").as_uri())
    modes = page.locator("#mode option").evaluate_all("(opts)=>opts.map(o=>o.value)")
    for mode in modes:
        page.select_option("#mode", mode)
        page.click("#step")
        assert page.locator("#tick").inner_text() == "1", mode
        assert page.locator("#metric").inner_text() != "NaN"
        first = page.locator("#metric").inner_text()
        page.click("#reset")
        page.click("#step")
        assert page.locator("#metric").inner_text() == first
    page.select_option("#mode", "circuit")
    page.click("#network summary")
    page.click("#run")
    page.wait_for_function("Number(document.querySelector('#tick').textContent)>2")
    page.click("#run")
    tick = page.locator("#tick").inner_text()
    page.wait_for_timeout(120)
    assert page.locator("#tick").inner_text() == tick
    with page.expect_download() as download:
        page.click("#save")
    trace = json.loads(Path(download.value.path()).read_text())
    assert len(trace["records"]) == int(tick)
    page.click("#theme")
    page.set_viewport_size({"width": 390, "height": 800})
    assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
    page.select_option("#study", "evolution")
    assert "independent search rmse" in page.locator("#metrics").inner_text()
    assert not errors, errors
    browser.close()
    print(
        f"Offline Chromium: {len(modes)} workspaces, replay, controls, export, verification, mobile layout passed"
    )
