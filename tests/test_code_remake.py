"""A first code remake built from the measured scenes: motion of the drawn layer, the redraw renderer and the
whole loop (write the project, match each scene, render, score)."""
import json

import cv2
import numpy as np

from rea_video.code_remake import code_remake
from rea_video.layers import scene_motion, step_motion
from rea_video.redraw import Redraw


def _dots(seed=0, cx=240, cy=135, n=1200):
    rng = np.random.default_rng(seed)
    img = np.zeros((270, 480, 3), np.uint8)
    for _ in range(n):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 1) ** 0.5
        cv2.circle(img, (int(cx + 90 * r * np.cos(a)), int(cy + 60 * r * np.sin(a))), 1, (230, 220, 60), -1)
    return img


def _video(path, frames, fps=8):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def test_step_motion_reads_zoom_rotation_and_pan():
    g = cv2.cvtColor(_dots(), cv2.COLOR_BGR2GRAY)
    M = cv2.getRotationMatrix2D((240, 135), -2, 1.03)  # OpenCV turns anticlockwise for positive angles
    M[0, 2] += 4
    s, rot, px, py, drift = step_motion(g, cv2.warpAffine(g, M, (480, 270)), g > 0)
    assert abs(s - 1.03) < 0.01 and abs(rot - 2) < 0.3 and abs(px - 4 / 480) < 0.003 and abs(py) < 0.003
    assert drift < 0.002
    m = scene_motion([[1.03, 2.0, 0.01, 0.0, 0.001]] * 4, 0.25)
    assert abs(m["rotate"] - 8.0) < 1e-6 and m["zoom"] > 0.1 and abs(m["pan"][0] - 0.04) < 1e-6
    assert scene_motion([], 0.25) == {}
    assert step_motion(g, g, np.zeros_like(g, bool)) is None  # nothing drawn to follow


def _spec(tmp_path, scenes):
    spec = {"size": [320, 180], "fps": 10, "duration": 4.0, "bloom": 0, "scenes": scenes}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return tmp_path / "spec.json"


def _points(tmp_path, name, xy):
    rows = [[x, y, 255, 255, 255, 1.0] for x, y in xy]
    (tmp_path / name).write_text(json.dumps({"points": rows}))
    return name


def test_redraw_places_particles_steadily_and_morphs_between_scenes(tmp_path):
    left = _points(tmp_path, "a.json", [(0.2 + 0.001 * i, 0.5) for i in range(50)])
    right = _points(tmp_path, "b.json", [(0.8 - 0.001 * i, 0.5) for i in range(50)])
    still = {"zoom": 0, "rotate": 0, "pan": [0, 0], "drift": 0, "twinkle": 0}
    spec = _spec(tmp_path, [
        {"id": "s1", "start": 0, "end": 2, "draw": "points", "points": left, "count": 50, "dot": 1, "gain": 1, "motion": still},
        {"id": "s2", "start": 2, "end": 4, "draw": "points", "points": right, "count": 50, "dot": 1, "gain": 1,
         "motion": still, "enter": "morph", "enter_s": 1.0}])
    rd = Redraw(spec)
    a, b = rd.frame(0.5), rd.frame(0.6)
    assert np.array_equal(a, b)  # no motion asked for: nothing flickers
    xcentre = lambda f: float(np.nonzero(f.max(axis=2) > 40)[1].mean()) / f.shape[1]  # noqa: E731
    assert abs(xcentre(a) - 0.22) < 0.03
    assert 0.35 < xcentre(rd.frame(2.5)) < 0.65  # half way through the morph
    assert abs(xcentre(rd.frame(3.5)) - 0.78) < 0.03
    assert rd.frame(5.0).max() == 0  # past the last scene


def test_code_remake_writes_a_project_that_lines_up_with_the_original(tmp_path):
    frames = [_dots(i % 2) for i in range(24)] + [_dots(i % 2, cx=360) for i in range(24)]
    video = _video(tmp_path / "v.mp4", frames)
    out = tmp_path / "out"
    out.mkdir()
    shot = {"index": 1, "start": 0.0, "end": 6.0, "deterministic": {"probs": {"code_render": 0.9},
                                                                    "facts": {"fine_dark": 0.9}}}
    b = {"source": {"path": str(video), "width": 480, "height": 270, "fps": 8, "duration": 6.0}, "shots": [shot]}
    (out / "breakdown.json").write_text(json.dumps(b))
    res = code_remake(out)
    spec = json.loads((out / "code" / "spec.json").read_text())
    assert [s["draw"] for s in spec["scenes"]] == ["points", "points"] and spec["scenes"][1]["enter"] == "morph"
    assert (out / "code" / "render.py").exists() and (out / "code" / "remake.mp4").exists()
    assert res["compare"]["score"] > 0.6
    spec["scenes"][0]["gain"] = 2.0  # an edit is kept on the next run
    (out / "code" / "spec.json").write_text(json.dumps(spec))
    assert code_remake(out, render=False)["kept"]
    assert json.loads((out / "code" / "spec.json").read_text())["scenes"][0]["gain"] == 2.0
