"""The drawn layer of rendered shots, measured for a code remake.

A code-rendered picture is a smooth background (flat colour, gradient, glow, vignette) with something drawn on it
(particles, lines, shapes, charts, text). The drawn layer is whatever stands out from the smooth background, away from
the ground colour, in any direction: light on dark, dark on light, or another colour of the same brightness.

`rea-video frames out/ref` splits rendered shots into scenes (the drawn layer rearranges without a cut), saves
full-size frames, a cut-out of the drawn layer and samples of it per scene, and measures the ground, the background
glow, the drawn colours and where the drawing sits. A remake can then place its graphics on the original's shapes
instead of redrawing them from a description.

`rea-video compare ref remake` scores a remake against the original frame by frame on the drawn layer (where it sits,
how well the shapes overlap, how much there is) and says what to change: timing, position, spread, amount, drawn
colour and ground. Each video's layer is taken against its own ground, so a remake on a different ground colour
compares without extra steps.
"""
from __future__ import annotations

import heapq
from pathlib import Path

import cv2
import numpy as np

from .features import DETAIL, halo_blur
from .i18n import T
from .media import extract_frame, iter_frames, video_info
from .util import read_json, write_json

WORK_W = 480  # measuring width; positions are written as 0-1 of the frame, so this only sets precision
GRID = (16, 9)  # cells of the layout map (x, y)
SCENE_WIN = 1.5  # seconds compared on each side of a candidate scene change; also the shortest scene kept
SCENE_MOVE = 0.25  # share of the drawn layer that has to move elsewhere in the frame for a new scene
SCENE_AMOUNT = 3.0  # or this many times more / less drawn in place
SCENE_COLOUR = 25.0  # or the drawn colour moving this far (Lab a, b)


def _hex(bgr) -> str:
    b, g, r = (int(round(float(v))) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def ground(bgr: np.ndarray) -> np.ndarray:
    """Background colour: the median of the frame border."""
    border = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
    return np.median(border, axis=0)


def contrast(bgr: np.ndarray) -> np.ndarray:
    """How far each pixel stands out from the smooth background, away from the ground: its colour distance to the
    ground colour minus the smooth background's (Lab). Light dots on a dark glow, dark ink on paper, a white road or a
    blue river on a pale map all stand out; the dark gaps between dots inside a glow (closer to the ground) do not."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = cv2.cvtColor(ground(bgr).astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    smooth = np.dstack([halo_blur(np.ascontiguousarray(lab[..., c])) for c in range(3)])
    return np.linalg.norm(lab - g, axis=2) - np.linalg.norm(smooth - g, axis=2)


def layer_mask(bgr: np.ndarray, text_boxes: list | None = None) -> np.ndarray:
    """The drawn layer. text_boxes (0-1 x, y, w, h) are blanked, so subtitles and labels are left out."""
    m = contrast(bgr) > DETAIL
    h, w = m.shape
    for x, y, bw, bh in text_boxes or []:
        pad = 0.01
        x0, y0 = int(max(0, x - pad) * w), int(max(0, y - pad) * h)
        x1, y1 = int(min(1, x + bw + pad) * w + 1), int(min(1, y + bh + pad) * h + 1)
        m[y0:y1, x0:x1] = False
    return m


def layout(mask: np.ndarray) -> np.ndarray:
    """Where the drawn layer sits: share of it in each grid cell."""
    d = cv2.resize(mask.astype(np.float32), GRID, interpolation=cv2.INTER_AREA)
    s = float(d.sum())
    return d / s if s > 0 else d


def _colours(pixels: np.ndarray, k: int = 3) -> list[dict]:
    """Main colours of a set of BGR pixels, as hex with their share."""
    if len(pixels) < 10:
        return []
    px = pixels.astype(np.float32)
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    k = min(k, len(np.unique(px, axis=0)))
    cv2.setRNGSeed(0)
    _, lbl, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2,
                                 cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lbl.ravel(), minlength=k) / len(lbl)
    return [{"hex": _hex(centers[i]), "share": round(float(share[i]), 2)} for i in np.argsort(-share) if share[i] >= 0.05]


def measure(bgr: np.ndarray, text_boxes: list | None = None) -> dict:
    """Ground, background glow (colour, centre, radius), drawn colours, how many separate pieces and how big, where
    the drawing sits and how much of the frame it covers. Positions are 0-1 of the frame."""
    h, w = bgr.shape[:2]
    gf = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = ground(bgr)
    out: dict = {"ground": _hex(g)}
    gl = float(cv2.cvtColor(g.astype(np.uint8)[None, None], cv2.COLOR_BGR2GRAY)[0, 0])
    lift = np.abs(halo_blur(gf) - gl)  # the smooth background's departure from the ground: glow, vignette, gradient
    if lift.max() > 6:
        on = lift > lift.max() * 0.5
        ys, xs = np.nonzero(on)
        wts = lift[on]
        peak = np.unravel_index(int(np.argmax(lift)), lift.shape)
        blur = cv2.resize(cv2.GaussianBlur(cv2.resize(bgr, (max(1, w // 4), max(1, h // 4))), (0, 0), max(h, w) / 80.0), (w, h))
        out["glow"] = {"colour": _hex(blur[peak]), "centre": [round(float((xs * wts).sum() / wts.sum()) / w, 3),
                                                             round(float((ys * wts).sum() / wts.sum()) / h, 3)],
                       "radius": round(float(np.sqrt(on.sum() / np.pi)) / w, 3)}
    m = layer_mask(bgr, text_boxes)
    n = int(m.sum())
    out["cover"] = round(n / float(h * w), 4)
    if n < 10:
        return out
    out["colours"] = _colours(bgr[m])
    ys, xs = np.nonzero(m)
    out["box"] = [round(float(np.percentile(xs, 2)) / w, 3), round(float(np.percentile(ys, 2)) / h, 3),
                  round(float(np.percentile(xs, 98)) / w, 3), round(float(np.percentile(ys, 98)) / h, 3)]
    out["centre"] = [round(float(xs.mean()) / w, 3), round(float(ys.mean()) / h, 3)]
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas):
        out["pieces"] = int(n_cc - 1)
        out["piece_area"] = round(float(np.median(areas)) / float(h * w), 6)  # median piece, share of the frame
    return out


def sample_points(bgr: np.ndarray, n: int = 4000, text_boxes: list | None = None) -> list[list[float]]:
    """Up to n pixels of the drawn layer as [x, y, r, g, b, a] (x, y 0-1; colour 0-255; a = how strongly the pixel
    stands out, 0-1), picked with probability by that strength: targets to place particles, dots or strokes on."""
    c = contrast(bgr)
    m = layer_mask(bgr, text_boxes)
    ys, xs = np.nonzero(m)
    if not len(xs):
        return []
    wts = c[ys, xs]
    pick = np.random.default_rng(0).choice(len(xs), size=min(n, len(xs)), replace=False, p=wts / wts.sum())
    h, w = c.shape
    top = float(wts.max()) or 1.0
    return [[round(float(xs[i] + 0.5) / w, 4), round(float(ys[i] + 0.5) / h, 4), *[int(v) for v in bgr[ys[i], xs[i]][::-1]],
             round(float(wts[i]) / top, 2)] for i in pick]


def cutout(bgr: np.ndarray, text_boxes: list | None = None) -> np.ndarray:
    """The drawn layer as an RGBA image (BGRA for OpenCV): the frame's colours, alpha by how strongly each pixel
    stands out from the smooth background; text left out."""
    c = np.clip(contrast(bgr) / 100.0, 0, 1)
    c[~cv2.dilate(layer_mask(bgr, text_boxes).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
    return np.dstack([bgr, (c * 255).astype(np.uint8)])


def _text_boxes(out: Path, b: dict) -> list[tuple[float, float, list]]:
    """Where text sits and when: every OCR read stored on the shots, plus the packaging timeline once it is written
    (decompose measures the layer before that file exists)."""
    boxes = [(float(it["t"]) - 0.6, float(it["t"]) + 0.6, it["bbox"]) for s in b.get("shots", [])
             for it in (s.get("text") or {}).get("items", []) if it.get("bbox") and it.get("t") is not None]
    tl = out / "packaging" / "timeline.json"
    if tl.exists():
        boxes += [(float(x["t0"]), float(x["t1"]), x["bbox"]) for x in read_json(tl).get("text", []) if x.get("bbox")]
    return boxes


def _boxes_at(boxes, t: float, until: float | None = None) -> list:
    """Text boxes on screen at t (or at any time in t..until): OCR misses text on some frames, so a scene blanks
    every box seen during it."""
    z = t if until is None else until
    return [b for t0, t1, b in boxes if t0 - 0.3 <= z and t <= t1 + 0.3]


def split_scenes(maps: np.ndarray, times: np.ndarray, start: float, end: float,
                 looks: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Scene changes inside one shot: where the layout before and after differs (share of the drawn layer that sits
    elsewhere) by more than SCENE_MOVE beyond how much it varies within each side, or where the drawing turns into
    something else in place: SCENE_AMOUNT times more or less of it, or its colour moves by SCENE_COLOUR. maps: one
    flattened layout per sampled time; looks: [log cover, a, b] of the drawn layer per sampled time (Lab)."""
    if len(times) < 2:
        return [(start, end)]
    step = float(np.median(np.diff(times)))
    win = max(1, int(round(SCENE_WIN / step)))
    score = np.zeros(len(times))
    for i in range(win, len(times) - win + 1):
        a, b = maps[i - win:i], maps[i:i + win]
        spread = 0.5 * (0.5 * np.abs(a - a.mean(0)).sum(1).mean() + 0.5 * np.abs(b - b.mean(0)).sum(1).mean())
        score[i] = (0.5 * np.abs(a.mean(0) - b.mean(0)).sum() - spread) / SCENE_MOVE
        if looks is not None:
            la, lb = np.median(looks[i - win:i], axis=0), np.median(looks[i:i + win], axis=0)
            score[i] = max(score[i], abs(float(lb[0] - la[0])) / np.log(SCENE_AMOUNT),
                           float(np.hypot(lb[1] - la[1], lb[2] - la[2])) / SCENE_COLOUR)
    cuts: list[float] = []
    for i in np.argsort(-score):
        if score[i] < 1.0:
            break
        t = float(times[i])
        if t - start < SCENE_WIN or end - t < SCENE_WIN or any(abs(t - c) < SCENE_WIN for c in cuts):
            continue
        lo, hi = max(0, i - win), min(len(score), i + win + 1)
        if score[i] >= score[lo:hi].max():
            cuts.append(t)
    edges = [start, *sorted(cuts), end]
    return list(zip(edges[:-1], edges[1:]))


def look(bgr: np.ndarray, m: np.ndarray) -> list[float]:
    """[log cover, a, b] of the drawn layer: how much is drawn and its mean colour (Lab), for split_scenes."""
    if m.sum() < 10:
        return [float(np.log(1e-4)), 0.0, 0.0]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[m].astype(np.float32)
    return [float(np.log(m.mean() + 1e-4)), float(lab[:, 1].mean() - 128), float(lab[:, 2].mean() - 128)]


def step_motion(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray) -> list[float] | None:
    """How the drawn layer moved from one sampled frame to the next (gray frames, mask = the drawn layer of prev):
    [scale, rotation in degrees, pan x, pan y, drift], pan as 0-1 of the frame for the layer's centre, drift = how far
    single points still move once the whole-layer move is taken out (0-1 of the frame width). None when too little
    is drawn to follow."""
    h, w = prev.shape
    pts = cv2.goodFeaturesToTrack(prev, 300, 0.01, 3, mask=cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)))
    if pts is None or len(pts) < 8:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(15, 15), maxLevel=2)
    ok = ok.ravel().astype(bool)
    p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
    if len(p0) < 8:
        return None
    A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None or inl is None or inl.mean() < 0.3:  # flickering dots: no one move most points agree on
        return None
    c = p0.mean(axis=0)
    moved = A[:, :2] @ c + A[:, 2]
    keep = inl.ravel().astype(bool)
    res = np.linalg.norm(p1[keep] - (p0[keep] @ A[:, :2].T + A[:, 2]), axis=1)
    return [round(float(np.hypot(A[0, 0], A[1, 0])), 5), round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 4),
            round(float(moved[0] - c[0]) / w, 5), round(float(moved[1] - c[1]) / h, 5), round(float(np.median(res)) / w, 5)]


def scene_motion(steps: list[list[float]], every: float) -> dict:
    """Whole-scene motion from the per-step moves: zoom (share per second, + is closer), rotation (degrees per
    second, + is clockwise on screen), pan (0-1 of the frame per second) and drift (how fast single points wander
    on their own, 0-1 of the frame width per second)."""
    if not steps:
        return {}
    a = np.array(steps, np.float64)
    dur = len(a) * every
    return {"zoom": round(float(np.exp(np.log(a[:, 0]).sum() / dur) - 1), 4), "rotate": round(float(a[:, 1].sum() / dur), 3),
            "pan": [round(float(a[:, 2].sum() / dur), 4), round(float(a[:, 3].sum() / dur), 4)],
            "drift": round(float(np.median(a[:, 4]) / every), 4)}


def shot_label(shot: dict) -> str | None:
    """The shot's current label: after review / correction when there is one, else the measured one."""
    base = (shot.get("base") or {}).get("label")
    if base:
        return base
    probs = (shot.get("deterministic") or {}).get("probs") or {}
    return max(probs, key=probs.get) if probs else None


def _clear(fdir: Path) -> None:
    """Drop what an earlier run wrote, so a rerun with fewer scenes leaves no stale files behind."""
    for p in [*fdir.glob("scene_*.jpg"), *fdir.glob("scene_*.png"), *(fdir / "points").glob("scene_*.json"), fdir / "scenes.json"]:
        p.unlink(missing_ok=True)
    for d in (fdir / "points", fdir):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def scene_frames(out: Path, every: float = 0.25, n_points: int = 4000, all_shots: bool = False, breakdown: dict | None = None) -> dict:
    """Scenes, full-size frames, a cut-out of the drawn layer, measurements and samples for the rendered shots of a
    breakdown (all shots with all_shots). Frames are measured as they are decoded, so memory does not grow with
    shot length."""
    out = Path(out)
    b = breakdown if breakdown is not None else read_json(out / "breakdown.json")
    video = (b.get("source") or {}).get("path")
    if not video or not Path(video).exists():
        raise FileNotFoundError(T("找不到原视频 {p}（breakdown.json 的 source.path）", "source video not found: {p} (source.path in breakdown.json)", p=video))
    fdir = out / "frames"
    _clear(fdir)
    shots = [s for s in b.get("shots", []) if all_shots or shot_label(s) == "code_render"]
    if not shots:
        return {"video": video, "every": every, "scenes": []}
    info = video_info(video)
    boxes = _text_boxes(out, b)
    (fdir / "points").mkdir(parents=True, exist_ok=True)
    full_w = min(1080, info.display_size[0] or 1080)
    scenes: list[dict] = []
    for s in shots:
        st, en = float(s["start"]), float(s["end"])
        maps, times, track, moves, looks = [], [], [], [], []
        prev = None
        for i, f in enumerate(iter_frames(video, info, WORK_W, start=st, duration=en - st, fps=1.0 / every)):
            t = st + i * every
            m = layer_mask(f, _boxes_at(boxes, t))
            maps.append(layout(m).ravel())
            looks.append(look(f, m))
            times.append(t)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                mv = step_motion(prev[0], g, prev[1])
                if mv:
                    moves.append([round(t, 2), *mv])
            prev = (g, m)
            ys, xs = np.nonzero(m)
            if len(xs) >= 10:  # how the drawing moves: centre, horizontal spread and cover over time
                track.append([round(t, 2), round(float(xs.mean()) / f.shape[1], 3), round(float(ys.mean()) / f.shape[0], 3),
                              round(float(xs.std()) / f.shape[1], 3), round(float(m.mean()), 4)])
        if not times:
            continue
        for a, z in split_scenes(np.array(maps), np.array(times), st, en, np.array(looks)):
            sid = f"scene_{len(scenes) + 1:02d}"
            mid = (a + z) / 2
            rec = {"id": sid, "shot": s.get("index"), "start": round(a, 2), "end": round(z, 2), "frames": {}}
            for tag, t in (("start", a + min(0.15, (z - a) / 4)), ("mid", mid), ("end", max(a, z - min(0.15, (z - a) / 4)))):
                p = fdir / f"{sid}_{tag}.jpg"
                extract_frame(video, t, p, full_w)
                rec["frames"][tag] = f"frames/{p.name}"
            full = cv2.imread(str(fdir / f"{sid}_mid.jpg"))
            if full is not None:
                in_scene = _boxes_at(boxes, a, z)
                cv2.imwrite(str(fdir / f"{sid}_layer.png"), cutout(full, in_scene))
                rec["frames"]["layer"] = f"frames/{sid}_layer.png"
                img = cv2.resize(full, (WORK_W, max(1, round(full.shape[0] * WORK_W / full.shape[1]))), interpolation=cv2.INTER_AREA)
                rec["measure"] = measure(img, in_scene)
                write_json(fdir / "points" / f"{sid}.json", {"time": round(mid, 2), "format": "x y r g b a",
                                                             "points": sample_points(img, n_points, in_scene)})
                rec["points"] = f"frames/points/{sid}.json"
            rec["track"] = [r for r in track if a <= r[0] < z]
            # a step that straddles a scene change is a rearrangement, not a move
            rec["motion"] = scene_motion([r[1:] for r in moves if a + every <= r[0] < z], every)
            rec["enter"] = "morph" if scenes and scenes[-1]["shot"] == s.get("index") else "cut"
            scenes.append(rec)
    res = {"video": video, "every": every, "track_format": "t cx cy spread cover", "scenes": scenes}
    write_json(fdir / "scenes.json", res)
    return res


def _lab_mean(lab: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    return lab[m].mean(axis=0) if m.any() else None


def _side(bgr: np.ndarray) -> dict:
    """One frame's drawn layer and what a remake can be told about it: where it sits, how spread out, how much,
    its mean colour and the ground's (Lab)."""
    m = layer_mask(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0  # OpenCV stores L as 0-255; colour distances below are in CIE units
    lab[..., 1:] -= 128.0
    g = cv2.cvtColor(ground(bgr).astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    g = np.array([g[0] * 100.0 / 255.0, g[1] - 128.0, g[2] - 128.0], np.float32)
    s = {"mask": m, "layout": layout(m), "cover": float(m.mean()), "ground": g, "colour": _lab_mean(lab, m)}
    if m.any():
        ys, xs = np.nonzero(m)
        h, w = m.shape
        s["centre"] = np.array([xs.mean() / w, ys.mean() / h])
        s["spread"] = np.array([xs.std() / w, ys.std() / h])
    return s


def _score(sa: dict, sb: dict, k: np.ndarray) -> dict:
    ma, mb = sa["mask"], sb["mask"]
    place = 1.0 - 0.5 * float(np.abs(sa["layout"] - sb["layout"]).sum()) if ma.any() and mb.any() else float(ma.any() == mb.any())
    ua, ub = cv2.dilate(ma.astype(np.uint8), k) > 0, cv2.dilate(mb.astype(np.uint8), k) > 0
    hit_a = float((ma & ub).sum()) / max(1, int(ma.sum()))
    hit_b = float((mb & ua).sum()) / max(1, int(mb.sum()))
    shape = 2 * hit_a * hit_b / (hit_a + hit_b) if hit_a + hit_b else (1.0 if not ma.any() and not mb.any() else 0.0)
    ca, cb = sa["cover"], sb["cover"]
    amount = min(ca, cb) / max(ca, cb) if max(ca, cb) > 0 else 1.0
    # equal weight on where it sits and on the shapes themselves; how much is drawn counts half as much
    return {"place": round(place, 3), "shape": round(shape, 3), "amount": round(amount, 3),
            "score": round(0.4 * place + 0.4 * shape + 0.2 * amount, 3)}


def _pair(sa: dict, sb: dict) -> dict:
    """Signed differences remake - original for one frame, None where one side has nothing drawn."""
    d: dict = {"cover": (sa["cover"], sb["cover"]), "ground": sb["ground"] - sa["ground"]}
    if sa["colour"] is not None and sb["colour"] is not None:
        d["colour"] = sb["colour"] - sa["colour"]
    if "centre" in sa and "centre" in sb:
        d["shift"] = sb["centre"] - sa["centre"]
        d["size"] = (sb["spread"] + 1e-6) / (sa["spread"] + 1e-6)
    return d


# How far off counts as visibly off, for the advice only (the score does not use these): a twentieth of the frame,
# a quarter bigger or smaller, a colour difference of 10 (CIE Lab, plainly different side by side).
OFF_SHIFT, OFF_RATIO, OFF_COLOUR = 0.05, 1.25, 10.0


def _advice(diffs: list[dict]) -> list[str]:
    """What to change in the remake over a stretch of frames, largest first."""
    if not diffs:
        return []
    tips: list[tuple[float, str]] = []
    sh = [d["shift"] for d in diffs if "shift" in d]
    if sh:
        dx, dy = np.median(np.array(sh), axis=0)
        for v, neg, pos in ((dx, ("偏左", "too far left"), ("偏右", "too far right")),
                            (dy, ("偏上", "too high"), ("偏下", "too low"))):
            if abs(v) >= OFF_SHIFT:
                tips.append((abs(v) / OFF_SHIFT, T(f"{(pos if v > 0 else neg)[0]} {abs(v) * 100:.0f}% 画面",
                                                     f"{(pos if v > 0 else neg)[1]} by {abs(v) * 100:.0f}% of the frame")))
        size = float(np.exp(np.median(np.log(np.array([d["size"].mean() for d in diffs if "size" in d])))))
        if max(size, 1 / size) >= OFF_RATIO:
            tips.append((max(size, 1 / size) / OFF_RATIO, T(f"铺开范围是原片的 {size:.1f} 倍", f"spread {size:.1f}x the original")))
    ca = float(np.mean([d["cover"][0] for d in diffs])); cb = float(np.mean([d["cover"][1] for d in diffs]))
    if ca > 0 and cb == 0:
        tips.append((9.0, T("画上去的东西没出来", "nothing drawn where the original has graphics")))
    elif cb > 0 and ca == 0:
        tips.append((9.0, T("原片这里没有画东西", "graphics where the original has none")))
    elif ca > 0:
        r = cb / ca
        if max(r, 1 / r) >= OFF_RATIO:
            tips.append((max(r, 1 / r) / OFF_RATIO, T(f"画上去的量是原片的 {r:.1f} 倍", f"{r:.1f}x as much drawn as the original")))
    for key, zh, en in (("colour", "前景颜色", "drawn colour"), ("ground", "底色", "ground")):
        cs = [d[key] for d in diffs if key in d]
        if not cs:
            continue
        dl, da, db = np.median(np.array(cs), axis=0)
        de = float(np.sqrt(dl * dl + da * da + db * db))
        if de >= OFF_COLOUR:
            how = []
            if abs(dl) >= OFF_COLOUR / 2:
                how.append(T("偏亮" if dl > 0 else "偏暗", "too light" if dl > 0 else "too dark"))
            hue = np.hypot(da, db)
            if hue >= OFF_COLOUR / 2:
                how.append(T("色相不对", "wrong hue"))
            tips.append((de / OFF_COLOUR, T(f"{zh}差 {de:.0f}（{'、'.join(how) or '偏色'}）",
                                            f"{en} off by {de:.0f} ({', '.join(how) or 'tinted'})")))
    return [t for _, t in sorted(tips, key=lambda x: -x[0])]


def _lag(la: list, lb: list, every: float, max_s: float = 3.0) -> tuple[float, float]:
    """Time offset of the remake that best lines its layout up with the original's, and how much place agreement
    it gains. Positive: the remake runs late."""
    n = len(la)
    best, base = 0, None
    gains = {}
    for L in range(-int(max_s / every), int(max_s / every) + 1):
        idx = [(i, i + L) for i in range(n) if 0 <= i + L < n]
        if len(idx) < max(4, n // 2):
            continue
        v = float(np.mean([1.0 - 0.5 * np.abs(la[i] - lb[j]).sum() for i, j in idx]))
        gains[L] = v
        if L == 0:
            base = v
    if base is None or not gains:
        return 0.0, 0.0
    best = max(gains, key=lambda L: (gains[L], -abs(L)))
    return round(best * every, 2), round(gains[best] - base, 3)


def compare(ref: str, remake: str, every: float = 0.5, out: Path | None = None, scenes: Path | None = None,
            worst: int = 8) -> dict:
    """Frame-by-frame score of a remake against the original on the drawn layer, with what to change: position,
    spread, amount, drawn colour, ground and timing. Both videos are decoded side by side and only the worst frames
    are kept, so memory does not grow with length."""
    ri, mi = video_info(ref), video_info(remake)
    dur = min(ri.duration, mi.duration)
    k = np.ones((5, 5), np.uint8)
    rows: list[dict] = []
    la: list = []
    lb: list = []
    keep: list = []  # min-heap on -score: the `worst` lowest-scoring frames with their pictures
    pairs = zip(iter_frames(ref, ri, WORK_W, start=0, duration=dur, fps=1.0 / every),
                iter_frames(remake, mi, WORK_W, start=0, duration=dur, fps=1.0 / every))
    for i, (fa, fb) in enumerate(pairs):
        fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        sa, sb = _side(fa), _side(fb)
        r = {"t": round(i * every, 2), **_score(sa, sb, k)}
        rows.append(r)
        del sa["mask"], sb["mask"]  # keep only the small numbers, so timing can be lined up afterwards
        la.append(sa)
        lb.append(sb)
        if out:
            item = (-r["score"], i, fa, fb)
            if len(keep) < worst:
                heapq.heappush(keep, item)
            elif item[0] > keep[0][0]:
                heapq.heapreplace(keep, item)
    if not rows:
        raise ValueError(T("没有可比较的帧", "no frames to compare"))
    res: dict = {"ref": ref, "remake": remake, "every": every,
                 **{k2: round(float(np.mean([r[k2] for r in rows])), 3) for k2 in ("score", "place", "shape")}}
    lag, gain = _lag([a["layout"] for a in la], [b["layout"] for b in lb], every)
    shift = 0
    if lag and gain >= 0.05:
        shift = int(round(lag / every))
        res["offset"] = {"seconds": lag, "place_gain": gain,
                         "advice": T(f"重做整体{'晚' if lag > 0 else '早'}了约 {abs(lag):g} 秒，先对齐时间；下面的建议已按对齐后算",
                                     f"the remake runs {abs(lag):g}s {'late' if lag > 0 else 'early'}; line up the timing first "
                                     "(the advice below is taken after lining it up)")}
    # the advice compares each original frame with the remake frame it lines up with
    diffs = [_pair(la[i], lb[i + shift]) if 0 <= i + shift < len(lb) else None for i in range(len(la))]
    res["advice"] = _advice([d for d in diffs if d])
    if scenes and Path(scenes).exists():
        per = []
        for sc in read_json(scenes).get("scenes", []):
            ix = [i for i, r in enumerate(rows) if sc["start"] <= r["t"] < sc["end"]]
            if ix:
                per.append({"id": sc["id"], "start": sc["start"], "end": sc["end"],
                            "score": round(float(np.mean([rows[i]["score"] for i in ix])), 3),
                            "shape": round(float(np.mean([rows[i]["shape"] for i in ix])), 3),
                            "advice": _advice([diffs[i] for i in ix if diffs[i]])})
        res["scenes"] = sorted(per, key=lambda x: x["score"])
    res["worst"] = sorted(r["t"] for r in sorted(rows, key=lambda r: r["score"])[:worst])
    for i in sorted(range(len(rows)), key=lambda i: rows[i]["score"])[:worst]:
        rows[i]["advice"] = _advice([diffs[i]] if diffs[i] else [])
    res["frames"] = rows
    if out:
        out.mkdir(parents=True, exist_ok=True)
        cells = []
        for _, i, fa, fb in sorted(keep, key=lambda x: x[1]):
            r = rows[i]
            row = np.hstack([fa, np.full((fa.shape[0], 6, 3), 128, np.uint8), fb])
            cv2.putText(row, f"{r['t']:.1f}s score {r['score']:.2f} (place {r['place']:.2f} shape {r['shape']:.2f})", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cells.append(row)
        if cells:
            cv2.imwrite(str(out / "compare_worst.jpg"), np.vstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 88])
            res["sheet"] = str(out / "compare_worst.jpg")
        write_json(out / "compare.json", res)
    return res


def drawn_colours(frames: list[np.ndarray]) -> list[str]:
    """Colours of the drawn layer of a shot (ground and smooth background left out), most common first."""
    px = [f[layer_mask(f)] for f in frames if f is not None]
    px = [p for p in px if len(p)]
    if not px:
        return []
    return [c["hex"] for c in _colours(np.concatenate(px), k=4)]
