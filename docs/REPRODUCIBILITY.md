# Reproduction

Reference run: Python 3.12.10, NumPy 2.4.3, seeds 0–7. From the repository root, run `python -m sonapse_recursion --seeds 8 --workers 3`, then `python scripts/build_demo.py`. For exact dependency selection use `python -m pip install numpy==2.4.3` with the reference Python version. Headless runs require no web service. Worker count is bounded from one to eight. Use one worker on a constrained laptop.

A SHA-256 digest of package source, Python version and NumPy version keys the trial cache with study and seed. Interruption loses at most uncommitted trials. Separate output directories preserve independent exports. CPU times depend on hardware and load. Numerical floating-point variation across BLAS implementations is possible. Individual host checkpoints store the complete trained projection, readout, recurrent state and virtual tick. Supply the same remaining input sequence on resumption.

Backend tests use unittest discovery. Optional browser tests use Python Playwright and a local Chromium executable selected by CHROMIUM_EXECUTABLE. Run `python tests/browser_smoke.py` for an offline real-browser check. Interactive exports contain the seed, workspace, condition and every retained input/output sample. A run pauses after 10,000 ticks, or after 20 generations for the recorded evolution study.

Use --profile educational, research, evolution, large-connectome, hardware-limited or headless. Explicit --study, --seeds and --workers override profile defaults. The large-connectome profile runs a sparse event-driven threshold network. Run python -m sonapse_recursion.replicate to freeze a pilot-driven selection and repeat two matched comparisons on fresh seeds. Its protocol rejects overlapping seed sets.

## Reproduce the anatomical import

From PowerShell, download the pinned CSV to a temporary file and run the importer:

```powershell
$sourceUrl = 'https://raw.githubusercontent.com/openworm/CElegansNeuroML/b36380a36d2a6dda0f03c946c433524b25ea2268/herm_full_edgelist.csv'
$inputCsv = Join-Path $env:TEMP 'recursion-openworm-edges.csv'
Invoke-WebRequest $sourceUrl -OutFile $inputCsv
python -m sonapse_recursion.import_connectome $inputCsv --source $sourceUrl --limit 25
```

The checked-in import report records the byte digest, node identifiers and edge count. Common scalar stimulation and a normalized, nonnegative weight matrix simplify the task. This model cannot infer physiological synaptic signs.

## Larger replication runs

The research-scale profile evaluates all registered studies across 32 seeds with four bounded workers. The broader replication export is generated with `python -m sonapse_recursion --profile research-scale --first-seed 1000 --out results/scale`. Its seeds are disjoint from the eight-seed reference suite and the earlier targeted replication. Use `--workers 1` on a constrained laptop. The local dashboard also supports batches of up to 64 seeds and reports the number of committed or cached trials.

The 65,536-node sparse study runs 512 ticks with 32 initial inputs and two outgoing edges per node. Time slots use a priority queue and simultaneous inputs aggregate before thresholding. The original 4,096-node study is still available under `--study sparse`. Bootstrap estimation uses the same seeded paired draws as the reference implementation and processes them in bounded chunks.
