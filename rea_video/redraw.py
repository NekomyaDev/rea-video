"""Redraw a code-rendered video from its measured scenes: a starting remake that already sits on the original's shapes.

`rea-video code-remake out/ref` copies this file into `out/ref/code/render.py` next to `spec.json`, so the remake
project runs on its own (Python with numpy and opencv, plus ffmpeg) and can be edited freely.

    python render.py                      # spec.json -> remake.mp4
    python render.py --frame 12.5         # one frame as frame_12.50.png, to check a scene quickly
    python render.py --start 10 --end 20  # a stretch only

spec.json holds everything that decides the picture; change values there rather than in this file:

- size, fps, duration; text: an ASS file burnt in on top (the measured subtitles and labels)
- bloom: how much the drawing glows (0 = none); bloom_size: how wide, as a share of the frame width
- scenes, one per stretch where the drawing keeps its arrangement:
  - start, end (seconds); ground: background colour; glow: soft background light (colour, centre 0-1, radius 0-1 of
    the width, strength)
  - draw: "points" (particles placed on the sampled points) or "cutout" (the cut-out picture of the drawn layer, moved)
  - points: JSON with rows [x, y, r, g, b, a] (x, y 0-1); a remake with new shapes swaps this file for its own
  - count: particles drawn; dot: particle radius in pixels; gain: brightness of the drawing
  - motion: zoom (share per second), rotate (degrees per second, clockwise), pan ([x, y] share of the frame per
    second), drift (how far each particle wanders, share of the width per second), twinkle (0-1)
  - enter: "morph" (particles travel from the previous scene's places over enter_s seconds) or "cut"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _bgr(hexs: str) -> np.ndarray:
    h = hexs.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32) / 255.0


def _zorder(xy: np.ndarray) -> np.ndarray:
    """Order points along a Z curve, so two shapes sorted this way pair up points that sit near each other."""
    q = np.clip((xy * 1023).astype(np.int64), 0, 1023)
    code = np.zeros(len(xy), np.int64)
    for b in range(10):
        code |= ((q[:, 0] >> b) & 1) << (2 * b) | ((q[:, 1] >> b) & 1) << (2 * b + 1)
    return np.argsort(code, kind="stable")


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Scene:
    def __init__(self, sc: dict, base: Path, W: int, H: int, seed: int):
        self.sc, self.W, self.H = sc, W, H
        self.start, self.end = float(sc["start"]), float(sc["end"])
        self.draw = sc.get("draw", "points")
        m = sc.get("motion") or {}
        self.zoom, self.rot = float(m.get("zoom", 0)), float(m.get("rotate", 0))
        self.pan = np.array(m.get("pan", [0, 0]), np.float32) * (W, H)
        self.drift = float(m.get("drift", 0)) * W
        self.twinkle = float(m.get("twinkle", 0.15))
        self.gain = float(sc.get("gain", 1.0))
        self.dot = float(sc.get("dot", 1.0))
        g = sc.get("glow") or {}
        self.ground = _bgr(sc.get("ground", "#000000"))
        self.bg = np.empty((H, W, 3), np.float32)
        self.bg[:] = self.ground
        if g.get("colour"):
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            cx, cy = g.get("centre", [0.5, 0.5])
            sigma = max(1.0, float(g.get("radius", 0.15)) * W / 1.177)  # radius is where the glow is half its peak
            fall = np.exp(-((xx - cx * W) ** 2 + (yy - cy * H) ** 2) / (2 * sigma * sigma))
            amp = np.clip(_bgr(g["colour"]) - self.ground, 0, None) * float(g.get("strength", 1.0))
            self.bg += fall[..., None] * amp
        self.centre = np.array(sc.get("centre", [0.5, 0.5]), np.float32) * (W, H)
        rng = np.random.default_rng(seed)
        if self.draw == "cutout":
            img = cv2.imread(str(base / sc["cutout"]), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            a = img[..., 3:4].astype(np.float32) / 255.0
            self.layer = img[..., :3].astype(np.float32) / 255.0 * a
            return
        rows = np.array(json.loads((base / sc["points"]).read_text(encoding="utf-8"))["points"], np.float32).reshape(-1, 6)
        n = int(sc.get("count", len(rows)))
        if not len(rows) or n <= 0:
            rows, n = np.zeros((0, 6), np.float32), 0
        idx = np.arange(n) % max(1, len(rows))
        pick = rows[idx]
        xy = pick[:, :2] * (W, H)
        extra = np.arange(n) >= len(rows)  # more particles than samples: spread the copies around their sample
        xy[extra] += rng.normal(0, max(1.5, 2.5 * self.dot), (int(extra.sum()), 2))
        order = _zorder(xy / (W, H))
        self.xy = xy[order].astype(np.float32)
        self.col = (pick[order][:, [4, 3, 2]] / 255.0 * pick[order][:, 5:6]).astype(np.float32)
        self.phase = rng.uniform(0, 2 * np.pi, (n, 3)).astype(np.float32)
        self.freq = rng.uniform(0.15, 0.45, (n, 2)).astype(np.float32)

    def _move(self, xy: np.ndarray, tau: float) -> np.ndarray:
        s = (1 + self.zoom) ** tau
        a = np.radians(self.rot * tau)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32) * s
        return (xy - self.centre) @ R.T + self.centre + self.pan * tau

    def particles(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        tau = t - self.start
        xy = self._move(self.xy, tau)
        if self.drift:
            amp = self.drift / (2 * np.pi * 0.3)  # wander at the measured speed around a fixed place, without flicker
            w = 2 * np.pi * self.freq * tau
            xy = xy + amp * np.stack([np.sin(w[:, 0] + self.phase[:, 0]), np.cos(w[:, 1] + self.phase[:, 1])], 1)
        col = self.col * (1 - self.twinkle + self.twinkle * (0.5 + 0.5 * np.sin(3 * tau + self.phase[:, 2:3])))
        return xy, col * self.gain

    def matrix(self, t: float) -> np.ndarray:
        tau = t - self.start
        s = (1 + self.zoom) ** tau
        M = cv2.getRotationMatrix2D(tuple(map(float, self.centre)), -self.rot * tau, s)
        M[:, 2] += self.pan * tau
        return M


class Redraw:
    def __init__(self, spec_path: Path):
        self.path = Path(spec_path)
        self.base = self.path.parent
        self.spec = json.loads(self.path.read_text(encoding="utf-8"))
        self.W, self.H = self.spec["size"]
        self.scenes = [Scene(sc, self.base, self.W, self.H, i) for i, sc in enumerate(self.spec["scenes"])]
        self.ground = np.zeros((self.H, self.W, 3), np.float32)

    def at(self, t: float) -> tuple[int, Scene | None]:
        for i, s in enumerate(self.scenes):
            if s.start <= t < s.end:
                return i, s
        return -1, None

    def _splat(self, buf: np.ndarray, xy: np.ndarray, col: np.ndarray) -> None:
        """Add each particle's colour at its sub-pixel place (shared between the four nearest pixels, so slow
        movement stays smooth)."""
        H, W = buf.shape[:2]
        x0, y0 = np.floor(xy[:, 0]), np.floor(xy[:, 1])
        fx, fy = xy[:, 0] - x0, xy[:, 1] - y0
        flat = buf.reshape(-1, 3)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).astype(np.int64), (y0 + dy).astype(np.int64)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            k = yi[ok] * W + xi[ok]
            for c in range(3):
                flat[:, c] += np.bincount(k, weights=col[ok, c] * wt[ok], minlength=H * W).astype(np.float32)

    def frame(self, t: float) -> np.ndarray:
        i, s = self.at(t)
        if s is None:
            return np.zeros((self.H, self.W, 3), np.uint8)
        prev = self.scenes[i - 1] if i > 0 else None
        enter = float(s.sc.get("enter_s", 0.6))
        mix = _ease((t - s.start) / enter) if (s.sc.get("enter") == "morph" and prev is not None and enter > 0) else 1.0
        bg = s.bg if mix >= 1 else prev.bg * (1 - mix) + s.bg * mix
        drawn = np.zeros((self.H, self.W, 3), np.float32)
        if s.draw == "cutout":
            drawn += cv2.warpAffine(s.layer, s.matrix(t), (self.W, self.H)) * s.gain * mix
            if mix < 1 and prev.draw == "cutout":
                drawn += cv2.warpAffine(prev.layer, prev.matrix(prev.end), (self.W, self.H)) * prev.gain * (1 - mix)
            dot = 0.0
        else:
            xy, col = s.particles(t)
            dot = s.dot
            if mix < 1 and prev.draw == "points" and len(prev.xy) and len(xy):
                pxy, pcol = prev.particles(prev.end)
                j = (np.arange(len(xy)) * len(pxy)) // len(xy)  # both are in Z order: neighbours travel to neighbours
                xy = pxy[j] * (1 - mix) + xy * mix
                col = pcol[j] * (1 - mix) + col * mix
                dot = prev.dot * (1 - mix) + s.dot * mix
            if len(xy):
                self._splat(drawn, xy, col)
                sig = max(0.6, 0.7 * dot)
                drawn = cv2.GaussianBlur(drawn, (0, 0), sig) * (2 * np.pi * sig * sig)
        bloom = float(self.spec.get("bloom", 0.8))
        if bloom > 0:
            small = cv2.resize(drawn, (self.W // 4, self.H // 4), interpolation=cv2.INTER_AREA)
            size = float(self.spec.get("bloom_size", 0.012)) * self.W / 4
            glow = cv2.GaussianBlur(small, (0, 0), size) + 0.5 * cv2.GaussianBlur(small, (0, 0), 3 * size)
            drawn = drawn + bloom * cv2.resize(glow, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        out = bg + drawn
        out = np.where(out > 0.85, 0.85 + 0.15 * np.tanh((out - 0.85) / 0.15), out)  # soft roll-off instead of hard clipping
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def _has_ass() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
        return " ass " in r.stdout
    except OSError:
        return False


def render(spec_path: Path, out: Path, start: float = 0.0, end: float | None = None, text: bool = True) -> Path:
    rd = Redraw(spec_path)
    fps = float(rd.spec.get("fps", 25))
    end = float(rd.spec["duration"]) if end is None else end
    vf = []
    ass = rd.spec.get("text")
    if text and ass and (rd.base / ass).exists():
        if _has_ass():
            vf = ["-vf", f"setpts=PTS+{start}/TB,ass={ass},setpts=PTS-STARTPTS"]
        else:
            print("ffmpeg has no ass filter (libass): text left out", file=sys.stderr)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{rd.W}x{rd.H}",
           "-r", f"{fps:g}", "-i", "-", *vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(Path(out).resolve())]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=rd.base)
    n = int(round((end - start) * fps))
    for k in range(n):
        p.stdin.write(rd.frame(start + k / fps).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render spec.json to a video, or one frame of it")
    ap.add_argument("spec", nargs="?", default=str(Path(__file__).with_name("spec.json")))
    ap.add_argument("-o", "--out")
    ap.add_argument("--frame", type=float, help="render the frame at this second to a PNG")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args(argv)
    spec = Path(a.spec)
    if a.frame is not None:
        out = Path(a.out or spec.with_name(f"frame_{a.frame:.2f}.png"))
        cv2.imwrite(str(out), Redraw(spec).frame(a.frame))
    else:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        out = render(spec, Path(a.out or spec.with_name("remake.mp4")), a.start, a.end, text=not a.no_text)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
