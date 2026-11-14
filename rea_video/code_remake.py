"""A runnable first remake of the code-rendered shots, built from the measured scenes.

`rea-video code-remake out/ref` writes `out/ref/code/`: `spec.json` (one entry per scene: ground, glow, the sampled
points or the cut-out, particle count and size, motion, how it enters), `render.py` (the renderer, a copy of
`redraw.py`) and `overlays.ass` (the measured text). Each scene's amount and brightness are matched to the original's
middle frame before the whole video is rendered and scored with `compare`. Whoever takes it further edits
spec.json, swaps the points for new shapes or rewrites render.py, then runs the command again (an existing
spec.json is kept; --reset starts over).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from . import redraw
from .i18n import T
from .layers import WORK_W, _boxes_at, _text_boxes, compare, layer_mask, scene_frames, shot_label
from .media import extract_frame
from .packaging import fit_ass_sizes, to_ass
from .plan import code_style
from .util import read_json, write_json

POINT_STYLES = ("particles", "lineart")  # drawn as particles on the sampled points; other styles move the cut-out


def _dot(m: dict, W: int) -> float:
    """Particle radius in output pixels from the median piece the layer splits into (pieces of a dense cloud or a
    line merge into one, so this is capped: more particles make up the rest)."""
    area = float(m.get("piece_area") or 0) * W * W * 9 / 16
    return round(float(np.clip(np.sqrt(area / np.pi) if area else 1.0, 0.7, 1.4)), 2)


def build_spec(out: Path) -> dict:
    """spec.json for the code-rendered scenes of a breakdown, before matching."""
    b = read_json(out / "breakdown.json")
    src = b.get("source") or {}
    fdir = out / "frames"
    if not (fdir / "scenes.json").exists():
        scene_frames(out, breakdown=b)
    scenes = read_json(fdir / "scenes.json").get("scenes", [])
    shots = {s.get("index"): s for s in b.get("shots", [])}
    W, H = int(src.get("width") or 960), int(src.get("height") or 540)
    spec: dict = {"size": [W, H], "fps": float(src.get("fps") or 25), "duration": float(src.get("duration") or 0),
                  "text": "overlays.ass" if any(not c.get("part_of_picture") for c in (b.get("packaging") or {}).get("text_cues", [])) else None,
                  "bloom": 0.5, "bloom_size": 0.012, "scenes": []}
    for sc in scenes:
        m = sc.get("measure") or {}
        style = code_style(shots.get(sc.get("shot")) or {})
        e = {"id": sc["id"], "shot": sc.get("shot"), "start": sc["start"], "end": sc["end"],
             "ground": m.get("ground") or "#000000", "centre": m.get("centre") or [0.5, 0.5]}
        if m.get("glow"):
            e["glow"] = {**m["glow"], "strength": 1.0}
        if style in POINT_STYLES and sc.get("points"):
            e.update(draw="points", points=f"../{sc['points']}", count=4000, dot=_dot(m, W), gain=1.0)
        elif (sc.get("frames") or {}).get("layer"):
            e.update(draw="cutout", cutout=f"../{sc['frames']['layer']}", gain=1.0)
        else:
            continue
        e["motion"] = {**{"zoom": 0, "rotate": 0, "pan": [0, 0], "drift": 0}, **(sc.get("motion") or {}),
                       "twinkle": 0.15 if e["draw"] == "points" else 0.0}
        e["enter"] = sc.get("enter", "cut")
        e["enter_s"] = round(min(0.8, (sc["end"] - sc["start"]) / 3), 2)
        spec["scenes"].append(e)
    return spec


def _side(bgr: np.ndarray, boxes: list) -> dict:
    """What the matching looks at: how much is drawn, how bright the drawing is and how bright the rest is."""
    m = layer_mask(bgr, boxes)
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    return {"cover": float(m.mean()), "lit": float(L[m].mean()) if m.any() else 0.0,
            "rest": float(L[~m].mean()) if (~m).any() else 0.0}


def match(out: Path, spec_path: Path, rounds: int = 5) -> list[dict]:
    """Match each scene's particle count, brightness and background glow to the original's middle frame: render
    that frame, measure it the same way as the original, scale, repeat."""
    b = read_json(out / "breakdown.json")
    video = (b.get("source") or {}).get("path")
    boxes = _text_boxes(out, b)
    spec = read_json(spec_path)
    tmp = spec_path.parent / "_match.jpg"
    log = []
    for e in spec["scenes"]:
        a, z = e["start"], e["end"]
        t = a + max(0.6 * (z - a), min(z - a, float(e.get("enter_s", 0)) + 0.05))  # past the morph in
        extract_frame(video, t, tmp, WORK_W)
        ref = cv2.imread(str(tmp))
        if ref is None:
            continue
        bx = _boxes_at(boxes, a, z)
        want = _side(ref, bx)
        for k in range(rounds):
            write_json(spec_path, spec)
            rd = redraw.Redraw(spec_path)
            rd.scenes = [s for s in rd.scenes if s.sc["id"] == e["id"]]
            got = _side(cv2.resize(rd.frame(t), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA), bx)
            if e["draw"] == "points" and got["cover"] > 0 and want["cover"] > 0:
                e["count"] = int(np.clip(e["count"] * (want["cover"] / got["cover"]) ** 0.9, 150, 40000))
            # amount first: brightness also moves how much stands out, so it is matched only once the count settles
            if k >= rounds // 2 and got["lit"] > 0 and want["lit"] > 0:
                e["gain"] = round(float(np.clip(e["gain"] * want["lit"] / got["lit"], 0.5, 4.0)), 3)
            if e.get("glow") and got["rest"] > 1 and want["rest"] > 1:
                e["glow"]["strength"] = round(float(np.clip(e["glow"]["strength"] * want["rest"] / got["rest"], 0.2, 3.0)), 3)
        log.append({"id": e["id"], "want": {k: round(v, 4) for k, v in want.items()},
                    "got": {k: round(v, 4) for k, v in got.items()}})
    tmp.unlink(missing_ok=True)
    write_json(spec_path, spec)
    return log


def code_remake(out: Path, reset: bool = False, render: bool = True, score: bool = True) -> dict:
    """Write (or keep) out/code, render remake.mp4 and score it against the original."""
    out = Path(out)
    b = read_json(out / "breakdown.json")
    if not any(shot_label(s) == "code_render" for s in b.get("shots", [])) and not (out / "frames" / "scenes.json").exists():
        raise ValueError(T("没有代码渲染镜头", "no code-rendered shots"))
    cdir = out / "code"
    cdir.mkdir(exist_ok=True)
    spec_path = cdir / "spec.json"
    res: dict = {"dir": str(cdir), "spec": str(spec_path)}
    shutil.copyfile(redraw.__file__, cdir / "render.py")
    cues = [c for c in (b.get("packaging") or {}).get("text_cues", []) if not c.get("part_of_picture")]
    if cues and (reset or not (cdir / "overlays.ass").exists()):
        src = b.get("source") or {}
        (cdir / "overlays.ass").write_text(to_ass(cues, int(src.get("width") or 960), int(src.get("height") or 540)), encoding="utf-8")
        fit_ass_sizes(cdir / "overlays.ass")
    if reset or not spec_path.exists():
        spec = build_spec(out)
        if not spec["scenes"]:
            raise ValueError(T("没有可重画的场景（先运行 rea-video frames）", "no scenes to redraw (run rea-video frames first)"))
        write_json(spec_path, spec)
        res["matched"] = match(out, spec_path)
    else:
        res["kept"] = True
    if render:
        video = redraw.render(spec_path, cdir / "remake.mp4")
        res["video"] = str(video)
        if score:
            ref = (b.get("source") or {}).get("path")
            cmp = compare(ref, str(video), out=cdir / "compare", scenes=out / "frames" / "scenes.json")
            res["compare"] = {k: cmp[k] for k in ("score", "place", "shape", "offset", "advice") if k in cmp}
            res["compare"]["lowest_scenes"] = cmp.get("scenes", [])[:5]
            res["compare"]["sheet"] = cmp.get("sheet")
    return res
