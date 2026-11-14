"""Video type and the hands-on remake steps, on hand-made breakdowns (no media needed)."""
from __future__ import annotations

import pytest

from rea_video import playbook
from rea_video.i18n import set_lang


def _b(mix: dict[str, float], fmt: str = "", sound: str = "", avg: float = 2.5, shots: int = 8, voice: float = 0.0,
       beat: bool = False, music: bool = True, subs: int = 0, chars: list | None = None, refs: list | None = None) -> dict:
    return {
        "summary": {"production_mix": [{"label": k, "ratio": v} for k, v in mix.items()], "avg_shot_s": avg, "shots": shots},
        "source": {"width": 1080, "height": 1920, "aspect_ratio": "9:16", "duration": shots * avg},
        "aspects": {"format": {"result": {"format": fmt}}} if fmt else {},
        "sound_sources": {"format_context": {"video_type": sound}},
        "audio": {"voice": {"present": voice > 0, "ratio": voice}, "music": {"present": music, "tempo_bpm": 120.0}, "beat_synced_edit": beat},
        "packaging": {"text_cues": [{"kind": "subtitle"}] * subs, "files": {"srt": "packaging/subtitles.srt"}},
        "production_plan": {"calls": 3, "model_name": "Seedance 2.0", "units": [{}, {}, {}]},
        "review_cast": {"characters": chars or []} if chars else {},
        "shots": [{"index": 1, "start": 0.0, "end": avg, "duration": avg, "material": {"references": refs or []}}],
    }


@pytest.mark.parametrize("b,want", [
    (_b({"live_action": 1.0}, fmt="talking_head", voice=0.8), "talking_head"),
    (_b({"live_action": 1.0}, fmt="desk"), "desk"),
    (_b({"live_action": 1.0}, fmt="vlog"), "vlog"),
    (_b({"ai_video": 1.0}, sound="music_mv"), "ai_mv"),
    (_b({"ai_video": 1.0}, refs=[{"kind": "product_photo"}]), "ai_product"),
    (_b({"ai_video": 1.0}, shots=20, voice=0.5), "ai_drama"),
    (_b({"ai_video": 1.0}, shots=5), "ai_clip"),
    (_b({"cg_3d": 1.0}), "animation"),
    (_b({"avatar": 1.0}), "avatar"),
    (_b({"code_render": 1.0}), "explainer"),
    (_b({"screen_capture": 1.0}), "screen"),
    (_b({"still_motion": 0.7, "static_hold": 0.3}), "slideshow"),
    (_b({"film_footage": 1.0}), "film"),
    (_b({"live_action": 1.0}, fmt="vlog", avg=0.9, shots=40), "beat"),
])
def test_type(b, want):
    assert playbook.kinds(b)[0][0] == want


def test_second_type_listed_when_it_covers_a_quarter():
    ks = playbook.kinds(_b({"live_action": 0.7, "code_render": 0.3}, fmt="talking_head", voice=0.6))
    assert [k for k, _ in ks] == ["talking_head", "explainer"]
    assert len(playbook.kinds(_b({"live_action": 0.85, "code_render": 0.15}, fmt="talking_head", voice=0.6))) == 1


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("kind", ["talking_head", "desk", "vlog", "live_drama", "ai_drama", "ai_product", "ai_mv", "avatar",
                                  "ai_clip", "animation", "film", "explainer", "screen", "slideshow", "beat"])
def test_every_kind_renders_steps(kind, lang):
    set_lang(lang)
    try:
        lines = playbook.steps(_b({"ai_video": 1.0}, voice=0.5, subs=3), kind)
        assert lines and all("{" not in x for x in lines)
        assert "rea-video remake" in lines[-1]
    finally:
        set_lang("zh")


def test_steps_only_carry_what_was_measured():
    set_lang("zh")
    b = _b({"live_action": 1.0}, fmt="talking_head", voice=0.8, shots=1, avg=16.0, subs=5)
    b["aspects"]["lighting"] = {"lines": ["打光：正面柔光（平光）。色温中性（约 5000K）"]}
    b["aspects"]["editing"] = {"lines": ["剪辑：每分钟 0.0 刀，实拍镜头平均 16.0s；机位/场景 1 个；同机位跳切 0 处、推近切 0 处"]}
    b["shooting_plan"] = {"gear": ["环形灯放在手机正后方略高处"]}
    md = "\n".join(playbook.md(b))
    assert "人物口播" in md and "subtitles.srt" in md and "一镜到底" in md
    assert "约 5000K" in md and "环形灯放在手机正后方略高处" in md
    assert "0 处" not in md and "机位/场景 1 个" not in md
    no_subs = "\n".join(playbook.md(_b({"live_action": 1.0}, fmt="talking_head", voice=0.8)))
    assert "subtitles.srt" not in no_subs and "布光" not in no_subs  # nothing measured, no step
