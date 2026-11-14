"""Rebuild a video on the original timeline from remake results.

- generated clips (capsule-cinema)  -> trimmed / padded to the original shot length
- still_motion / static_hold shots  -> rebuilt locally from one image using the measured
                                       zoom/pan (deterministic, no model call)
- everything else                    -> original footage kept (marked in the compare video)
- with overlays=True, the measured subtitles / 花字 / titles are laid back on over the replaced shots only (kept
  original footage already carries them)
Outputs remake.mp4 (original audio) and compare.mp4 (original | remake side by side)."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2

from .util import LOG, read_json, read_image


def _ff(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, cwd=cwd)


def render_still_motion(image: Path, out: Path, duration: float, size: tuple[int, int], fps: float,
                        zoom_per_s: float = 0.0, pan_px_per_s: float = 0.0, pan_dir_deg: float = 0.0,
                        analysis_width: int = 480) -> Path:
    """Re-create a Ken Burns move: per-frame similarity transform about the frame centre,
    with the zoom rate and pan speed measured during decomposition."""
    img = read_image(image)
    if img is None:
        raise FileNotFoundError(image)
    w, h = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    n = max(1, int(round(duration * fps)))
    px_scale = w / float(analysis_width)
    vx = math.cos(math.radians(pan_dir_deg)) * pan_px_per_s * px_scale
    vy = math.sin(math.radians(pan_dir_deg)) * pan_px_per_s * px_scale
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    # the measured motion maps frame t -> t+1; start slightly zoomed in when zooming out so edges never show
    base = 1.0 + max(0.0, -zoom_per_s * duration) + 0.02 + (abs(vx) + abs(vy)) * duration / max(w, h)
    for i in range(n):
        t = i / fps
        s = base * (1.0 + zoom_per_s * t)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        M[0, 2] += vx * t - vx * duration / 2
        M[1, 2] += vy * t - vy * duration / 2
        frame = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out


MAX_SPEEDUP = 3.0   # a generated clip is retimed to the original shot length, up to this much faster
MAX_SLOWDOWN = 1.25  # and up to this much slower; past that the remainder is trimmed / held


def fit_speed(clip_s: float, duration: float) -> float:
    """Playback speed that fits a generated clip into the original shot. Models have a minimum length
    (often 4-5 s), so a 1.2 s shot comes back as 4 s of the same action: playing only its first 1.2 s
    cuts the action off and slows the edit, so the whole clip is sped up instead (within limits)."""
    if clip_s <= 0 or duration <= 0:
        return 1.0
    return max(1 / MAX_SLOWDOWN, min(MAX_SPEEDUP, clip_s / duration))


def _clip_seconds(src: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True, check=True)
        return float(r.stdout.strip() or 0)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def _fit_clip(src: Path, out: Path, duration: float, size: tuple[int, int], fps: float) -> Path:
    w, h = size
    speed = fit_speed(_clip_seconds(src), duration)
    vf = (f"setpts=PTS/{speed:.4f},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
          f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},setpts=PTS-STARTPTS")
    _ff(["-i", str(src), "-an", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)])
    return out


def _original_segment(src: Path, out: Path, start: float, duration: float, size: tuple[int, int], fps: float) -> Path:
    w, h = size
    _ff(["-ss", f"{start:.4f}", "-i", str(src), "-t", f"{duration:.4f}", "-an",
         "-vf", f"scale={w}:{h},fps={fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)])
    return out


def replaced_spans(b: dict, legend: list[tuple]) -> list[tuple[float, float]]:
    """Time spans of shots no longer showing the original picture, adjacent ones merged."""
    shots = {sh["index"]: sh for sh in b["shots"]}
    spans: list[list[float]] = []
    for i, how in legend:
        if how in ("original", "code"):  # both already carry their text
            continue
        a, z = shots[i]["start"], shots[i]["end"]
        if spans and abs(spans[-1][1] - a) < 1e-3:
            spans[-1][1] = z
        else:
            spans.append([a, z])
    return [(a, z) for a, z in spans]


def overlay_cues(b: dict, spans: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """Text layers to lay back on, clipped to the replaced spans, and the stickers there that need an asset by hand.
    A layer cut by a span edge loses its entrance animation, so it does not pop in again mid-line."""
    from .packaging import ANCHOR_KINDS, cjk_count, speech_lines

    pack = b.get("packaging") or {}
    speech = {c["id"] for c in speech_lines(pack.get("text_cues", [])) if c.get("id")}
    cues, stickers = [], []
    for c in pack.get("text_cues", []):
        if c["kind"] == "scene" or c.get("part_of_picture"):
            continue
        # spoken lines and styled text go back on; other OCR reads only when they hold real words, not "8uXrH"-like noise
        if not (c.get("id") in speech or (c["kind"] in ANCHOR_KINDS and c["kind"] != "label") or cjk_count(c.get("text", "")) >= 2):
            continue
        for a, z in spans:
            lo, hi = max(c["t0"], a), min(c["t1"], z)
            if hi - lo >= 0.1:
                cues.append({**c, "t0": lo, "t1": hi, **({"anim_in": None} if lo > c["t0"] + 1e-3 else {}),
                             **({"anim_out": None} if hi < c["t1"] - 1e-3 else {})})
    for st in pack.get("stickers", []):
        if not st.get("in_picture") and st.get("confidence") != "low" and any(min(st["t1"], z) - max(st["t0"], a) >= 0.1 for a, z in spans):
            stickers.append({k: st.get(k) for k in ("id", "kind_zh", "what", "t0", "t1", "bbox", "image")})
    return cues, stickers


def assemble(breakdown_path: Path, results_path: Path, out_dir: Path | None = None, overlays: bool = False,
             code_video: Path | None = None, extra: dict | None = None) -> str:
    """code_video: a remake of the code-rendered shots on the original timeline (`code-remake`), cut in for them.
    extra: more {shot: {"video": path}} results on top of results_path's (later ones win)."""
    b = read_json(breakdown_path)
    bdir = breakdown_path.parent
    jobs_path = results_path.with_name("jobs.json")
    jobs = {j["shot"]: j for j in read_json(jobs_path)["jobs"]} if jobs_path.is_file() else {}
    results = read_json(results_path).get("results", {}) if results_path.is_file() else {}
    results = {**results, **{str(k): v for k, v in (extra or {}).items()}}
    out_dir = out_dir or results_path.parent
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    src = Path(b["source"]["path"])
    w, h = b["source"]["width"], b["source"]["height"]
    if b["source"].get("rotation") in (90, 270):
        w, h = h, w
    w, h = w - w % 2, h - h % 2
    fps = float(b["source"]["fps"]) if b["source"]["fps"] and b["source"]["fps"] <= 60 else 30.0
    parts, legend = [], []
    for sh in b["shots"]:
        dur = sh["duration"]
        seg = seg_dir / f"shot_{int(sh['index']):03d}.mp4"  # 不用 JSON 里的 id 拼路径，防 ../
        res = results.get(str(sh["index"])) or {}
        job = jobs.get(sh["index"], {})
        if res.get("video") and Path(res["video"]).is_file():
            _fit_clip(Path(res["video"]), seg, dur, (w, h), fps)
            legend.append((sh["index"], "generated"))
        elif code_video and sh["base"]["label"] == "code_render" and Path(code_video).is_file():
            _original_segment(Path(code_video), seg, sh["start"], dur, (w, h), fps)
            legend.append((sh["index"], "code"))
        elif job.get("executor") == "local_ffmpeg" or sh["base"]["label"] in ("still_motion", "static_hold"):
            m = sh["replication"].get("local", {})
            img = bdir / sh["assets"]["first_frame"]
            render_still_motion(img, seg, dur, (w, h), fps, m.get("zoom_per_s", 0.0), m.get("pan_px_per_s", 0.0),
                                m.get("pan_dir_deg", 0.0), m.get("analysis_width", 480))
            legend.append((sh["index"], "local_still"))
        else:
            _original_segment(src, seg, sh["start"], dur, (w, h), fps)
            legend.append((sh["index"], "original"))
        parts.append(seg)
    lst = seg_dir / "concat.txt"
    lst.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in parts), encoding="utf-8")
    silent = out_dir / "remake_silent.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent)])
    overlay_log: dict = {}
    if overlays:
        from .packaging import fit_ass_sizes, to_ass

        cues, stickers = overlay_cues(b, replaced_spans(b, legend))
        overlay_log = {"text": len(cues), "stickers_to_add": stickers}
        if cues:
            (out_dir / "overlays_remake.ass").write_text(to_ass(cues, w, h), encoding="utf-8")
            fit_ass_sizes(out_dir / "overlays_remake.ass")
            try:  # relative name + cwd: no filter-string escaping of the path
                _ff(["-i", silent.name, "-vf", "ass=overlays_remake.ass", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                     "remake_overlaid.mp4"], cwd=out_dir)
                silent = out_dir / "remake_overlaid.mp4"
                overlay_log["ass"] = str(out_dir / "overlays_remake.ass")
            except subprocess.CalledProcessError as exc:  # ffmpeg built without libass
                overlay_log["error"] = f"ffmpeg ass filter failed ({exc.returncode}); overlays_remake.ass can be burned in another editor"
                LOG.warning(overlay_log["error"])
    final = out_dir / "remake.mp4"
    if b["source"].get("has_audio"):
        _ff(["-i", str(silent), "-i", str(src), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)])
    else:
        final = silent
    compare = out_dir / "compare.mp4"
    ch = 720 if h >= 720 else h - h % 2
    label = ",drawtext=text='{t}':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.5"
    for with_text in (True, False):  # drawtext needs libfreetype; fall back to plain hstack
        la, lb = (label.format(t="original"), label.format(t="remake")) if with_text else ("", "")
        try:
            _ff(["-i", str(src), "-i", str(final), "-filter_complex",
                 f"[0:v]scale=-2:{ch}{la}[a];[1:v]scale=-2:{ch}{lb}[b];[a][b]hstack=inputs=2[v]",
                 "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-shortest", str(compare)])
            break
        except subprocess.CalledProcessError:
            if not with_text:
                raise
    log = {"legend": legend, "remake": str(final), "compare": str(compare), **({"overlays": overlay_log} if overlays else {})}
    (out_dir / "assemble_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("remake -> %s ; compare -> %s", final, compare)
    return json.dumps(log, ensure_ascii=False)


GENERATED = ("ai_video", "cg_3d", "anim_2d", "stop_motion", "avatar")


def _next_step(sh: dict) -> str:
    """What still has to happen for a shot that is showing the original."""
    from .i18n import T

    lab = sh["base"]["label"]
    need = (sh.get("replication") or {}).get("user_assets") or []
    if lab in GENERATED:
        return T("生成：用报告「生成单元」的提示词在网页上手动生成，或 remake-plan + remake-run（计费，先 --dry-run）",
                 "generate: paste the report's generation-unit prompt into a video model, or remake-plan + remake-run (paid; --dry-run first)")
    if lab == "live_action":
        return T("拍摄：按报告的拍摄方案拍", "shoot it: follow the report's shooting plan") + (
            T("（要用到：{a}）", " (needs: {a})", a="、".join(str(a) for a in need)) if need else "")
    if lab == "screen_capture":
        return T("录屏：按原片的操作重录一遍", "record the screen: redo the same steps")
    if lab == "film_footage":
        return T("影视片段：找到原片对应片段（有版权），或换成自己拍的", "film clip: find it in the source (rights apply) or shoot your own")
    if lab == "code_render":
        return T("代码渲染：rea-video code-remake 没跑成，先单独跑它看报错", "code render: code-remake did not run; run it alone to see why")
    return T("提供素材", "provide the footage")


def remake(out: Path, code: bool = True, overlays: bool = True) -> dict:
    """The whole remake in one pass, as far as it can go without paid generation or the user's own footage:
    code-rendered shots are redrawn (code-remake), stills are rebuilt from the measured move, clips already
    generated (remake/results.json from remake-run) or supplied (remake/mine.json, {"results": {"<shot>": {"video": path}}})
    are cut in, everything else keeps the original as a stand-in. Writes remake/remake.mp4, remake/compare.mp4 and
    remake/status.md (per shot: how it is made now and what is still to do). Run it again whenever a clip comes in."""
    from .i18n import T
    from .util import write_json

    out = Path(out)
    b = read_json(out / "breakdown.json")
    rdir = out / "remake"
    rdir.mkdir(exist_ok=True)
    res: dict = {}
    code_video = out / "code" / "remake.mp4"
    if code and any(sh["base"]["label"] == "code_render" for sh in b["shots"]):
        from .code_remake import code_remake

        try:
            if not code_video.is_file():
                cr = code_remake(out)
                res["code_score"] = (cr.get("compare") or {}).get("score")
        except (ValueError, OSError, RuntimeError) as e:
            res["code_error"] = str(e)
    mine = rdir / "mine.json"
    extra = read_json(mine).get("results", {}) if mine.is_file() else {}
    log = json.loads(assemble(out / "breakdown.json", rdir / "results.json", rdir, overlays=overlays,
                              code_video=code_video if code_video.is_file() else None, extra=extra))
    how = dict(log["legend"])
    HOW = {"generated": T("生成 / 你的素材", "generated / your footage"), "code": T("代码重画", "redrawn in code"),
           "local_still": T("静图按测得的运动重建", "still rebuilt with the measured move"), "original": T("原片占位", "original as stand-in")}
    rows, todo = [], []
    for sh in b["shots"]:
        h = how.get(sh["index"], "original")
        nxt = _next_step(sh) if h == "original" else ""
        rows.append({"shot": sh["index"], "start": sh["start"], "end": sh["end"], "label": sh["base"]["label"], "now": h, "next": nxt})
        if nxt:
            todo.append(sh["index"])
    done = sum(r["end"] - r["start"] for r in rows if r["now"] != "original")
    total = sum(r["end"] - r["start"] for r in rows) or 1.0
    L = [T("# 复刻进度", "# Remake status"), "",
         T(f"已经替换 {done / total:.0%} 时长；成片 `remake/{Path(log['remake']).name}`，左右对比 `remake/compare.mp4`。",
           f"{done / total:.0%} of the running time replaced; video `remake/{Path(log['remake']).name}`, side by side `remake/compare.mp4`."), "",
         T("拿到新的镜头视频（生成的或用户自己的）就写进 `remake/mine.json`：`{\"results\": {\"镜头号\": {\"video\": \"绝对路径\"}}}`，再跑一次 `rea-video remake <目录>`。",
           "When a clip comes in (generated or the user's own), add it to `remake/mine.json`: `{\"results\": {\"<shot>\": {\"video\": \"<absolute path>\"}}}`, then run `rea-video remake <folder>` again."),
         "", T("| 镜头 | 时间 | 画面类型 | 现在 | 还要做 |", "| Shot | Time | Picture | Now | To do |"), "|---|---|---|---|---|"]
    from .classify import LABEL_ZH

    for r in rows:
        L.append(f"| #{r['shot']} | {r['start']:.1f}–{r['end']:.1f}s | {LABEL_ZH.get(r['label'], r['label'])} | {HOW[r['now']]} | {r['next'] or '—'} |")
    (rdir / "status.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(rdir / "status.json", {"replaced_share": round(done / total, 3), "shots": rows})
    res.update({"remake": log["remake"], "compare": log["compare"], "status": str(rdir / "status.md"),
                "replaced_share": round(done / total, 3), "to_do": todo})
    return res
