"""Optional end-to-end test of a real, local browser-launched benchmark."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as output:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sonapse_recursion.dashboard",
            "--out",
            output,
            "--port",
            "0",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        url = server.stdout.readline().strip()
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=os.environ.get("CHROMIUM_EXECUTABLE")
            )
            page = browser.new_page()
            page.goto(url)
            page.select_option("#study", "logic")
            page.fill("#seedCount", "32")
            page.click("#launch")
            page.wait_for_function(
                "document.querySelector('#job').textContent.includes('Completed 32 trials')"
            )
            assert (Path(output) / "report.json").exists()
            assert "truth table accuracy" in page.locator("#metrics").inner_text()
            assert page.locator("#cohort").input_value() == "live"
            page.select_option("#cohort", "scale")
            page.select_option("#study", "sparse_65k")
            assert "active nodes" in page.locator("#metrics").inner_text()
            browser.close()
            print(
                "Live dashboard launched 32 trials, persisted and displayed their results"
            )
    finally:
        server.terminate()
        server.wait(timeout=10)
