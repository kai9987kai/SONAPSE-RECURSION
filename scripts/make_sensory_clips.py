"""Generate the three original synthetic stimuli. Optional build dependency: imageio-ffmpeg."""

from pathlib import Path
import subprocess
import imageio_ffmpeg
import numpy as np

root = Path(__file__).resolve().parents[1] / "web" / "stimuli"
root.mkdir(exist_ok=True)
y, x = np.mgrid[:96, :160]
for name in ["orbit", "ripple", "crossing"]:
    frames = []
    for tick in range(72):
        angle = tick * 2 * np.pi / 72
        if name == "orbit":
            mask = (x - 80 - 50 * np.cos(angle)) ** 2 + (
                y - 48 - 27 * np.sin(angle)
            ) ** 2 < 14**2
            rgb = np.where(mask[..., None], [255, 174, 59], [23, 48, 59])
        elif name == "ripple":
            wave = 0.5 + 0.5 * np.cos(np.hypot(x - 72, y - 45) * 0.18 - angle * 2)
            rgb = np.array([15, 41, 63]) + wave[..., None] * [42, 150, 141]
        else:
            rgb = np.tile([33, 36, 60], (96, 160, 1))
            rgb[(abs(x - 80 - 54 * np.sin(angle)) < 13) & (abs(y - 37) < 15)] = [
                231,
                104,
                113,
            ]
            rgb[(abs(x - 80 + 54 * np.sin(angle)) < 13) & (abs(y - 59) < 15)] = [
                112,
                175,
                233,
            ]
        frames.append(rgb.astype("uint8").tobytes())
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "160x96",
            "-r",
            "12",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libvpx",
            "-b:v",
            "120k",
            str(root / f"{name}.webm"),
        ],
        input=b"".join(frames),
        check=True,
    )
