"""Build five original 480x288 periodic visual stimuli for the offline laboratory."""

from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np


def frame(name, x, y, phase):
    z = x + 1j * y
    if name in ("julia", "mandelbrot"):
        if name == "julia":
            c = -0.73 + 0.16j + 0.07 * np.exp(1j * phase)
        else:
            c = z * (0.28 + 0.12 * np.cos(phase)) - 0.7 + 0.1j
            z = np.zeros_like(c)
        escape = np.zeros(x.shape)
        for n in range(54):
            active = np.abs(z) < 4
            z = np.where(active, z * z + c, z)
            escape += active
        field = np.log1p(escape) * 2.1
    elif name == "carpet":
        u, v = (x + 0.28 * np.cos(phase)) * 0.45, (y + 0.28 * np.sin(phase)) * 0.45
        depth = np.zeros(x.shape)
        active = np.ones(x.shape, dtype=bool)
        for _ in range(5):
            u, v = (u % 1) * 3, (v % 1) * 3
            active &= ~((u.astype(int) == 1) & (v.astype(int) == 1))
            depth += active
        field = depth * 0.85 + 0.18 * np.sin(phase)
    elif name == "interference":
        field = np.cos(10 * x + phase) * np.sin(8 * y - phase) + np.sin(
            7 * np.abs(z) - phase
        )
    else:
        field = np.sin(9 * x + 2 * np.sin(5 * y + phase) + phase) + np.cos(
            4 * y - phase
        )
    channels = [0.52 + 0.46 * np.cos(field + p) for p in (0.2, 2.25, 4.2)]
    return np.uint8(np.stack(channels, axis=-1) * 255)


def main():
    root = Path(__file__).resolve().parents[1] / "web/stimuli"
    y, x = np.mgrid[-1:1:288j, -1.67:1.67:480j]
    for name in ("julia", "mandelbrot", "carpet", "interference", "ribbons"):
        args = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "480x288",
            "-r",
            "12",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libvpx",
            "-b:v",
            "500k",
            str(root / f"{name}.webm"),
        ]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            for tick in range(48):
                encoder.stdin.write(frame(name, x, y, tick * 2 * np.pi / 48).tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError(f"Encoding {name} failed")


if __name__ == "__main__":
    main()
