# Multistream imagination

Open the local dashboard at **Multistream imagination**. Click **Add example** repeatedly to combine clips. **Add webcam**, **Add video files** and **Add stream** can run alongside them. URL input expects a direct browser-playable media URL; a remote host must permit cross-origin canvas access. Uploaded files and webcam frames are processed in the browser. The URL source contacts the host you select.

The eight examples are Orbit, Ripple, Crossing, Julia set, Mandelbrot zoom, Sierpiński carpet, Wave interference and Flowing ribbons. The five newer clips are 480 × 288, encoded at 12 fps, with four-second loops. They can be regenerated with `python scripts/make_fractal_clips.py`. The offline dashboard embeds the complete collection.

## Observation and memory

Sources occupy separate tiles with their aspect ratios preserved. At each completed sampling tick, the studio encodes a composite frame. **Also remember each individual feed** additionally encodes the separate source views at the selected sample resolution when mixing inputs. An IndexedDB transaction commits each batch of images and metadata together. Turning the option off preserves composite capture and the existing archive. Source changes, loops and resolution changes start new episodes. Individual remove controls release only the selected input; **Stop studio** releases all inputs.

Sampling, storage and synthesis support 160 × 96, 320 × 192, 640 × 384, 1280 × 720 and 1920 × 1080. **Save imagined PNG** exports the generated canvas at its actual resolution. The default is 320 × 192. Larger dimensions affect real image samples and prediction computation. The browser stores compressed images and a smaller 12 × 8 RGB descriptor, 96 recurrent activations and three trained host outputs for each frame. Compression can change fine detail. Sparse recurrent activity is calculated with 288 directed synapses. Input images from lower-resolution sources retain their original detail limits.

New studio observations are automatically retained in this browser origin with no application-level frame-count cap. A disk/quota failure pauses capture and leaves successfully committed batches available. The browser profile, its storage quota, available RAM and the user's storage-clearing settings constrain retention. Existing saved classic memories and observations from the classic workspace in the same page enter the studio context. The original workspace retains its original Save, Clear, export and replay behavior.

## What a prediction uses

Every indexed observation contributes its spatial color descriptor to a similarity-weighted context image. Distances use color features and the trained host response. The three most relevant observations that have a successor in the same episode supply full-resolution motion/detail estimates. Single-frame episodes still contribute color context. The compact index is scanned in full. A small decoded-image cache limits full-resolution working memory. Longer archives increase retrieval cost. The live counter shows how many memories contributed to the last completed synthesis and how many inputs are active.

The **Imagination horizon** advances 1, 4 or 8 synthetic steps. Intermediate images and recurrent states feed subsequent steps. Live error measures the one-step prediction against the next observation and compares it with holding the previous scene. Multi-step display does not change which prediction is scored. **Imagine from memory** releases camera and video inputs, begins from a chosen episode and rolls forward using the entire available archive. Generated images stay outside the observation archive. The algorithm synthesizes motion and color from experience. It can blur, repeat or drift over a long rollout.

The capture loop skips timer callbacks while an earlier update is in progress. With many inputs, a large archive, or long high-resolution rollouts, the effective observation rate depends on device performance. No fixed real-time throughput is claimed.

## Export and validation

**Export studio memories** downloads a JSON snapshot containing compressed images, episode provenance, full descriptor/state vectors and source labels. The file size grows with the archive. `python scripts/visual_replay.py studio-memory.json` verifies the sensory dynamics and trained host response independently from the exported precompression descriptors. The same command accepts the original classic format. **Clear studio archive** removes the studio's saved observations after confirmation. The classic memory control remains separate.

The automated browser check uses synthetic camera input, three simultaneously active fractal videos, two selected files and a direct video URL. It checks real 640 × 384 synthesis, source removal, export, independent host replay, a 1920 × 1080 PNG download, persistence and autonomous recall from an archive above the earlier 240-frame limit. Run `python tests/studio_smoke.py` against the local dashboard with the Playwright test environment.
