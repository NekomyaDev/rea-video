"""What kind of video this is, and the hands-on steps to make one like it.

The kind decides which steps there are and in what order (a talking head is shot, voiced, cut and subtitled; an
AI short needs references, first frames and generation units; a beat montage starts from the music). The content
of every step comes from this breakdown (measured framing, light, cutting, sound, packaging, the generation plan)
or from what this tool produces (files and commands). A step with no data behind it is left out.

The kind comes from what is already measured: the share of each picture class, the live-action format (talking
head / desk / vlog), the sound type (song, voice-over, location sound), beat-synced cutting and the cast.
"""
from __future__ import annotations

import re

from . import entities, narrative, performance
from .i18n import LangDict, T, model_name

NAME_ZH = LangDict({
    "talking_head": "人物口播", "desk": "桌拍 / 开箱 / 带货", "vlog": "vlog 随拍", "live_drama": "实拍剧情",
    "ai_drama": "AI 剧情短片", "ai_product": "AI 商品广告", "ai_mv": "AI 音乐 MV", "avatar": "数字人口播",
    "ai_clip": "AI 短片段（萌宠 / 奇观 / 单场景）", "animation": "3D / 动画",
    "film": "影视解说 / 混剪", "explainer": "动效科普 / 数据动画", "screen": "录屏教程 / 软件演示",
    "slideshow": "图片推拉成片（图文 / 故事 / 推文）", "beat": "卡点混剪",
}, en={
    "talking_head": "talking head", "desk": "tabletop / unboxing / product", "vlog": "vlog", "live_drama": "live-action skit",
    "ai_drama": "AI story short", "ai_product": "AI product ad", "ai_mv": "AI music video", "avatar": "digital-avatar talking head",
    "ai_clip": "short AI clip (pets / spectacle / single scene)", "animation": "3D / animation",
    "film": "film commentary / film-clip montage", "explainer": "motion-graphics explainer", "screen": "screen tutorial / software demo",
    "slideshow": "stills with pan and zoom (photo story)", "beat": "beat-synced montage",
})

_LIVE = ("live_action",)
_AI = ("ai_video",)
_ANIM = ("cg_3d", "anim_2d", "stop_motion")
_STILL = ("still_motion", "static_hold")
_LIVE_KINDS = ("talking_head", "desk", "vlog", "live_drama")
_GENERATED_KINDS = ("ai_drama", "ai_product", "ai_mv", "avatar", "ai_clip", "animation")


def _shares(b: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in (b.get("summary") or {}).get("production_mix") or []:
        out[m["label"]] = out.get(m["label"], 0.0) + float(m.get("ratio") or 0)
    return out


def _format(b: dict) -> str:
    return (((b.get("aspects") or {}).get("format") or {}).get("result") or {}).get("format") or ""


def _sound_type(b: dict) -> str:
    return ((b.get("sound_sources") or {}).get("format_context") or {}).get("video_type") or ""


def _has_product(b: dict, cast: dict) -> bool:
    if any(c.get("kind") == "product" for c in cast.get("characters") or []):
        return True
    return any(a.get("kind") in ("product_photo", "logo") for r in b.get("shots", []) for a in (r.get("material") or {}).get("references", []))


def _people(cast: dict) -> int:
    return sum(1 for c in cast.get("characters") or [] if c.get("kind") in ("person", "animal", "cartoon", "avatar"))


def _group(shares: dict[str, float]) -> list[tuple[str, float]]:
    g = {"live": sum(shares.get(k, 0) for k in _LIVE), "ai": sum(shares.get(k, 0) for k in _AI),
         "avatar": shares.get("avatar", 0), "anim": sum(shares.get(k, 0) for k in _ANIM),
         "code": shares.get("code_render", 0), "screen": shares.get("screen_capture", 0),
         "still": sum(shares.get(k, 0) for k in _STILL), "film": shares.get("film_footage", 0)}
    return sorted(((k, v) for k, v in g.items() if v > 0), key=lambda x: -x[1])


def _type_of_group(b: dict, grp: str, cast: dict) -> str:
    s, audio = b.get("summary") or {}, b.get("audio") or {}
    if grp == "live":
        fmt = _format(b)
        if fmt == "talking_head":
            return "talking_head"
        if fmt in ("desk", "product"):
            return "desk"
        talkers = (b.get("speakers") or {}).get("count") or 0
        return "live_drama" if _people(cast) >= 2 or (talkers >= 2 and s.get("shots", 0) >= 6) else "vlog"
    if grp in ("ai", "anim"):
        if _sound_type(b) == "music_mv":
            return "ai_mv"
        if _has_product(b, cast):
            return "ai_product"
        if grp == "anim":
            return "animation"
        long_story = float((b.get("source") or {}).get("duration") or 0) >= 40 or _people(cast) >= 2
        talks = bool((audio.get("voice") or {}).get("present"))
        return "ai_drama" if long_story and (talks or s.get("shots", 0) >= 8) else "ai_clip"
    return {"avatar": "avatar", "code": "explainer", "screen": "screen", "still": "slideshow", "film": "film"}[grp]


def kinds(b: dict) -> list[tuple[str, float]]:
    """[(type, share)], the main type first; a second type is listed when it covers at least a quarter of the time."""
    shares = _shares(b)
    groups = _group(shares)
    if not groups:
        return []
    cast = entities.cast(b)
    s, audio = b.get("summary") or {}, b.get("audio") or {}
    voice = float((audio.get("voice") or {}).get("ratio") or 0) if (audio.get("voice") or {}).get("present") else 0.0
    avg, music = float(s.get("avg_shot_s") or 9), bool((audio.get("music") or {}).get("present"))
    if _sound_type(b) != "music_mv" and ((audio.get("beat_synced_edit") and avg <= 1.6 and voice < 0.3)
                                         or (music and avg <= 1.2 and voice < 0.1)):
        return [("beat", 1.0)]
    out = [(_type_of_group(b, groups[0][0], cast), groups[0][1])]
    for g, v in groups[1:]:
        t = _type_of_group(b, g, cast)
        if v >= 0.25 and t != out[0][0]:
            out.append((t, v))
            break
    return out


def _after_colon(x: str) -> str:
    for sep in ("：", ": "):
        if sep in x:
            return x.split(sep, 1)[1].strip()
    return x.strip()


_UNKNOWN = ("未测出", "not measured")


def _aspect(b: dict, key: str) -> str:
    lines = ((b.get("aspects") or {}).get(key) or {}).get("lines") or []
    if not lines:
        return ""
    sep = T("；", "; ")
    parts = [p for p in _after_colon(lines[0]).split(sep) if p.strip() and not any(u in p for u in _UNKNOWN)]
    return sep.join(parts)


def _src(b: dict, k: str) -> str:
    v = (b.get("sound_sources") or {}).get(k) or {}
    if not v.get("source_zh"):
        return ""
    sure = v.get("source") != "unknown" and (v.get("probs") or {}).get(v.get("leaning"), 0) >= 0.6
    z = v["source_zh"]
    return z + ("" if sure or T("待确认", "to confirm") in z else T("（倾向，待确认）", " (leaning, to confirm)"))


def _cut(b: dict) -> str:
    s, src = b.get("summary") or {}, b.get("source") or {}
    n, dur = int(s.get("shots") or 0), float(src.get("duration") or 0)
    if n <= 1:
        return T("原片一镜到底（{d:.0f} 秒）", "the original is a single {d:.0f} s take", d=dur)
    return T("原片 {n} 个镜头、平均 {a:.1f} 秒一刀", "the original has {n} shots at {a:.1f}s each", n=n, a=float(s.get("avg_shot_s") or 0))


# Every step below is built from this video's breakdown or from what this tool produces (files, commands).
# None of them carries a rule of thumb from elsewhere: when the data for a step is missing, the step is left out.

def _content(b: dict) -> str | None:
    beats = narrative.beats(b)
    lines = entities.script(b)
    if beats:
        return T("**写内容**：原片分 {k} 段（见「内容与结构」的叙事线），照这个顺序写自己的版本，每段对应的秒数照原片",
                 "**Write the content**: the original runs in {k} steps (the narrative line under “Content and structure”); write yours in the same order and timing",
                 k=len(beats))
    n = (b.get("summary") or {}).get("shots") or 0
    if n <= 1 and not lines:
        if any(c.get("kind") == "subtitle" for c in (b.get("packaging") or {}).get("text_cues") or []):
            return T("**写内容**：原片一镜到底，说了什么看「剪辑包装层」里的字幕原文，写自己的版本",
                     "**Write the content**: the original is one take; what it says is in the subtitle text under “Packaging layer”; write your own")
        return T("**写内容**：原片一镜到底，画面内容见分镜表，写自己的版本", "**Write the content**: the original is one take, described in the shot list; write your own")
    if lines:
        return T("**写内容**：原片台词 {k} 句（「声音」里的脚本表），{n} 个镜头，逐镜写自己的画面和台词",
                 "**Write the content**: the original has {k} lines (script table under “Sound”) over {n} shots; write your picture and line for each shot", k=len(lines), n=n)
    return T("**写内容**：照分镜表的 {n} 个镜头逐镜写自己的画面", "**Write the content**: one picture of your own for each of the {n} shots in the shot list", n=n)


def _material(b: dict, live: bool, refs: bool) -> str | None:
    s = b.get("summary") or {}
    rm = s.get("real_material") or {}
    kinds: list[str] = []
    for k in rm.get("kinds") or {}:
        k = k.split("（")[0].split(" (")[0].strip()
        if k not in kinds and not (live and k in ("实拍视频素材", "live footage")):
            kinds.append(k)
    cast = entities.cast(b)
    who = [c.get("name") for c in cast.get("characters") or [] if c.get("name")]
    out = []
    if kinds:
        out.append(T("画面里的真实素材（{k}）要拿到原文件，清单见「要准备的素材」", "the real material in the picture ({k}) needs the original files, listed under “What to prepare”",
                     k=T("、", ", ").join(kinds)))
    if who and refs:
        out.append(T("每个角色准备一张参考图，所有镜头都用同一张：{w}", "one reference image per character, the same one in every shot: {w}", w=T("、", ", ").join(who[:6])))
    if s.get("references") and refs:
        out.append(T("商品、Logo 等参考图见「复刻要准备的参考素材」", "product and logo references under “Reference material to prepare”"))
    return (T("**备素材**：", "**Collect material**: ") + T("；", "; ").join(out)) if out else None


def _shoot(b: dict) -> list[str]:
    plan, sp = b.get("production_plan") or {}, b.get("shooting_plan") or {}
    shoot = plan.get("shoot") or []
    if not shoot and not sp:
        return []
    L = []
    if shoot:
        st = sum(len(x.get("setups") or []) for x in shoot)
        if len(shoot) > 1 or st > 1:
            L.append(T("**排拍摄**：{sc} 个场景、{st} 个机位，同一场景的镜头一起拍，顺序和每个机位要拍哪些镜头见下方「拍摄」",
                       "**Plan the shoot**: {sc} scenes, {st} camera setups; shoot each scene in one go, the order and the shots per setup are under “Shoot” below",
                       sc=len(shoot), st=st))
    look = [x for x in (_aspect(b, "framing"), _aspect(b, "camera")) if x]
    if look:
        L.append(T("**机位**：{x}；画幅 {ar}", "**Camera**: {x}; frame {ar}", x=T("；", "; ").join(look), ar=(b.get("source") or {}).get("aspect_ratio") or ""))
    light = _aspect(b, "lighting")
    if light:
        gear = sp.get("gear") or []
        L.append(T("**布光**：原片测得 {l}", "**Light**: measured on the original: {l}", l=light)
                 + (T("。可以这样打：{g}", ". One way to get it: {g}", g=T("；", "; ").join(gear)) if gear else ""))
    grade = _aspect(b, "grade")
    if grade:
        L.append(T("**调色**：{g}", "**Grade**: {g}", g=grade))
    return L


def _units(b: dict) -> list[dict]:
    return [u for u in (b.get("production_plan") or {}).get("units") or [] if u.get("kind", "generate") == "generate"]


def _generate(b: dict) -> list[str]:
    """Prompts first, then first frames, then the calls: the order a generation actually needs its inputs in."""
    plan, units = b.get("production_plan") or {}, _units(b)
    if not units:
        return []
    L = []
    if any("缺提示词" in (u.get("prompt") or "") or "missing prompt" in (u.get("prompt") or "").lower() for u in units):
        L.append(T("**补提示词**：这次没开看图，生成单元里的镜头描述是空的；先让 agent 跑 `rea-video review` 看图补写",
                   "**Fill in the prompts**: this run had no image reading, so the shot descriptions in the units are empty; have the agent run `rea-video review` first"))
    own = [u for u in units if any(i.get("kind") == "first_frame" and i.get("need") == "yes" and "shot_" in (i.get("source") or "") for i in u.get("inputs") or [])]
    chained = [u for u in units if any(i.get("kind") == "first_frame" and i.get("need") == "yes" and "shot_" not in (i.get("source") or "") for i in u.get("inputs") or [])]
    if own:
        L.append(T("**首帧图**：{k} 个单元从原片首帧起步（`shots/shot_XXX_first.jpg`）；换人物、商品或画风时，先照单元里的首帧提示词重画这张图",
                   "**First frames**: {k} units start from the original's first frame (`shots/shot_XXX_first.jpg`); to change the person, product or look, redraw it from the unit's first-frame prompt first",
                   k=len(own))
                 + (T("；另外 {c} 个单元接在上一个单元的尾帧上，要按顺序生成", "; {c} more units start on the previous unit's last frame, so generate them in order", c=len(chained)) if chained else ""))
    elif chained:
        L.append(T("**按顺序生成**：{c} 个单元接在上一个单元的尾帧上", "**Generate in order**: {c} units start on the previous unit's last frame", c=len(chained)))
    native = ((b.get("sound_sources") or {}).get("voice") or {}).get("source") == "ai_video_native"
    L.append(T("**生成**：{u} 个生成单元、共 {c} 次 {m}，每个单元的提示词、要喂的图和衔接方式见下方「AI 生成」",
               "**Generate**: {u} generation units, {c} calls to {m}; each unit's prompt, input images and how it joins the last are under “AI generation” below",
               u=len(units), c=plan.get("calls") or len(units), m=model_name(plan.get("model_name")) or T("视频模型", "a video model"))
             + (T("；原片的对白是随画面一起生成的，台词写在提示词里", "; the original's dialogue was generated with the picture, so the lines go in the prompt") if native else "")
             + T("。可以手动贴到模型网页上生成，也可以让 agent 用 capsule-cinema 自动跑（按调用计费，先试跑看任务和费用）",
                 ". Paste them into the model's web page by hand, or have the agent run them through capsule-cinema (paid per call; it does a dry run first to show the jobs and cost)"))
    return L


def _shoot_few(b: dict) -> str | None:
    """A handful of live shots inside a generated video: point at the plan instead of measuring the whole film as if it were shot."""
    shoot = (b.get("production_plan") or {}).get("shoot") or []
    if not shoot:
        return None
    n = sum(len(x.get("shots") or []) for x in shoot)
    return T("**实拍镜头**：{n} 个镜头要自己拍，机位和布光见下方「拍摄」", "**Live shots**: {n} shots to film yourself; camera and light under “Shoot” below", n=n)


def _avatar(b: dict) -> str | None:
    av = (b.get("production_plan") or {}).get("avatar") or []
    if not av:
        return None
    return T("**数字人**：{k} 段数字人口播，形象、音色和每段台词见下方「AI 生成」", "**Avatar**: {k} avatar segments; face, voice and lines per segment under “AI generation” below", k=len(av))


def _code(b: dict) -> str | None:
    if not (b.get("production_plan") or {}).get("code"):
        return None
    return T("**动效画面**：`rea-video code-remake <报告目录>` 会按测得的版式和动效写出一个能改的工程和第一版成片，改文字和数据后重跑；每段画面元素和动效时间见下方「代码合成」",
             "**Motion graphics**: `rea-video code-remake <report dir>` writes an editable project and a first cut from the measured layout and motion; change the text and data and rerun; elements and timing per scene under “Code build” below")


def _local(b: dict) -> str | None:
    if not (b.get("production_plan") or {}).get("local"):
        return None
    return T("**静图推拉**：换成自己的图，推拉方向和速度用下方「本地重建」里测得的数；`rea-video remake` 会按这些数自动做出来",
             "**Stills with moves**: swap in your images and use the measured direction and speed under “Local rebuild” below; `rea-video remake` builds them from those numbers")


def _footage(b: dict) -> str | None:
    if not (b.get("production_plan") or {}).get("footage"):
        return None
    return T("**影视片段**：要用到的片段和时长见下方「影视原片」；用别人的影视画面要先确认有没有使用权",
             "**Film clips**: the clips and lengths are under “Film / TV source” below; check that you have the right to use them")


def _voice(b: dict) -> str | None:
    if not ((b.get("audio") or {}).get("voice") or {}).get("present"):
        return None
    if ((b.get("sound_sources") or {}).get("voice") or {}).get("source") == "ai_video_native" and _units(b):
        return None  # generated with the picture: the generate step says so
    src = _src(b, "voice")
    k = len(entities.script(b))
    acted = performance.table(b)[0]
    bits = [T("原片是 {s}", "the original is {s}", s=src)] if src else []
    if k:
        bits.append(T("台词 {k} 句在「声音」的脚本表里，每句的语速和停顿也在那里", "{k} lines in the script table under “Sound”, with the pace and pauses of each", k=k))
    if acted:
        bits.append(T("语气照「表演与情绪」", "delivery per “Acting and emotion”"))
    return (T("**人声**：", "**Voice**: ") + T("；", "; ").join(bits)) if bits else None


def _edit_detail(b: dict) -> list[str]:
    """The measured editing facts that say something: no zero counts, no pause-cutting without speech."""
    e = _aspect(b, "editing")
    talk = ((b.get("audio") or {}).get("voice") or {}).get("present")
    single = int((b.get("summary") or {}).get("shots") or 0) <= 1
    out = []
    for part in e.split(T("；", "; "))[1:]:
        p = part.strip()
        if not p or re.search(r"(?<![\d.])0(\.0)? ?(处|个|%|cuts?|setups?|scenes?)", p) or ((T("说话停顿", "pause") in p) and not talk):
            continue
        if p in (T("没有实拍镜头", "no live-action shots"),) or (single and re.search(r"\b1 ?(个|setup|scene)", p)):
            continue
        out.append(p)
    return out


def _edit(b: dict) -> str:
    detail = _edit_detail(b)
    trans = {k: v for k, v in ((b.get("summary") or {}).get("transitions") or {}).items() if k not in ("start", "cut")}
    return (T("**剪辑**：按分镜表的顺序和时长剪，{c}", "**Edit**: follow the order and lengths in the shot list; {c}", c=_cut(b))
            + (T("；", "; ") + T("；", "; ").join(detail) if detail else "")
            + (T("；有 {n} 处不是硬切，位置见下方「剪辑」", "; {n} transitions are not hard cuts, see “Edit” below", n=sum(trans.values())) if trans else ""))


def _packaging(b: dict) -> str | None:
    pk = (b.get("summary") or {}).get("packaging") or []
    files = (b.get("packaging") or {}).get("files") or {}
    subs = [c for c in (b.get("packaging") or {}).get("text_cues") or [] if c.get("kind") == "subtitle"]
    if not pk and not subs:
        return None
    first = next((x for x in pk if x.startswith((T("字幕", "Subtitle"), T("花字", "Styled"), T("标题", "Title")))), "")
    if not first and not subs:
        return T("**字幕和花字**：时间和样式见「剪辑包装层」", "**Subtitles and styled text**: timing and style under “Packaging layer”")
    out = T("**字幕和花字**：{p}", "**Subtitles and styled text**: {p}", p=first) if first else T("**字幕和花字**：", "**Subtitles and styled text**: ")
    if subs and files.get("srt"):
        out += (T("；", "; ") if first else "") + T("字幕时间轴存成了 `{f}`（标准 SRT），能导入 SRT 的剪辑软件都能直接用，换成自己的话即可",
                 "the subtitle timing is saved as `{f}` (standard SRT), which any editor that imports SRT can use; replace the words",
                 f=files["srt"])
    return out + T("；其余花字、贴纸的时间和样式见「剪辑包装层」", "; the other styled text and stickers, with timing and style, under “Packaging layer”")


def _music(b: dict) -> str | None:
    audio = b.get("audio") or {}
    m = audio.get("music") or {}
    if not m.get("present"):
        return None
    bits = []
    src = _src(b, "music")
    if src:
        bits.append(T("来源{s}", "source: {s}", s=src))
    bpm = audio.get("tempo_bpm") or m.get("tempo_bpm")
    if bpm:
        bits.append(T("约 {b:.0f} BPM", "about {b:.0f} BPM", b=float(bpm)))
    key = m.get("key") or {}
    if key.get("tonic") and float(key.get("confidence") or 0) >= 0.6:
        bits.append(T("{t} {m}", "{t} {m}", t=key["tonic"], m=T("大调", "major") if key.get("mode") == "major" else T("小调", "minor")))
    if audio.get("beat_synced_edit"):
        bits.append(T("切镜卡在节拍上", "cuts land on the beat"))
    return T("**配乐**：{x}；找同速度、同情绪的曲子，风格和段落见「声音」", "**Music**: {x}; find a track at the same tempo and mood, style and sections under “Sound”",
             x=T("，", ", ").join(bits) or T("有配乐", "music present"))


def _sfx(b: dict) -> str | None:
    sfx = [e for e in (b.get("audio_plan") or {}).get("sfx_merged", []) if not e.get("in_scene")]
    if not sfx:
        return None
    return T("**音效**：{k} 处，时间点和各自卡在哪个画面事件上见「声音」，每处都能先试听", "**Sound effects**: {k}, with the time and the picture event each one hits under “Sound”; each can be played first",
             k=len(sfx))


def _assemble() -> str:
    return T("**拼回去对照**：拍好或生成好的镜头交给 agent，用 `rea-video remake` 拼回原时间线，还没做的镜头先用原片占位，会出一条和原片左右并排的对比片",
             "**Put it together and compare**: hand the shots you made to the agent; `rea-video remake` places them on the original timeline, keeps the original where yours are missing, and renders a side-by-side comparison with the original")


# Kinds with their own order. "content" and "material" stand for the per-video steps filled in by steps().
_ORDER_SPEC: dict[str, list] = {
    "beat": [_music, "material", _edit, _packaging],
    "film": ["material", _footage, "content", _voice, _edit, _packaging, _music],
    "explainer": ["content", "material", _voice, _code, _local, _packaging, _music, _sfx],
    "screen": ["content", "material", _voice, _edit, _packaging, _music],
    "slideshow": ["content", "material", _voice, _local, _edit, _packaging, _music],
    "avatar": ["content", "material", _voice, _avatar, _edit, _packaging, _music],
    "ai_mv": [_music, "content", "material", _generate, _shoot_few, _local, _edit, _packaging, _sfx],  # the picture follows the song
}


def steps(b: dict, kind: str, also: str | None = None) -> list[str]:
    """The steps for this video in the order its kind is made; only steps with data behind them appear.
    `also`: a second kind covering a good part of the time; then every making step with data is listed."""
    live = kind in _LIVE_KINDS and not also
    content, mat = _content(b), _material(b, live, refs=kind in _GENERATED_KINDS or bool(also))
    spec = None if also else _ORDER_SPEC.get(kind)
    order = [{"content": content, "material": mat}.get(x, x) if isinstance(x, str) else x for x in spec] if spec else None
    if order is None:
        if live:
            order = [content, mat, _shoot, _voice, _edit, _packaging, _music, _sfx]
        else:  # a few live shots inside a generated video get one pointer; a real live share gets the full shoot steps
            shoot = _shoot if also in _LIVE_KINDS or kind in _LIVE_KINDS else _shoot_few
            order = [content, mat, _generate, _avatar, shoot, _code, _local, _voice, _edit, _packaging, _music, _sfx]
    out: list[str] = []
    for x in order:
        got = x(b) if callable(x) else x
        for line in got if isinstance(got, list) else [got]:
            if line and line not in out:
                out.append(line)
    return out + [_assemble()]


def md(b: dict) -> list[str]:
    """The hands-on part of the remake section: the kind of video, then numbered steps built from this breakdown."""
    ks = kinds(b)
    if not ks:
        return []
    main = ks[0][0]
    L = [T("这条属于「{k}」。下面每一步的数都是从这条原片测出来的，照着做，做完拿来对照。",
           "This is a {k}. Every number in the steps below was measured on the original; follow them and check yours against it.", k=NAME_ZH[main])]
    if len(ks) > 1:
        L[0] += T("另有 {p:.0%} 的时长是「{k}」，步骤里一起列了。", " {p:.0%} of the time is a {k}; its steps are included.", p=ks[1][1], k=NAME_ZH[ks[1][0]])
    L.append("")
    L += [f"{i}. {x}" for i, x in enumerate(steps(b, main, also=ks[1][0] if len(ks) > 1 else None), 1)]
    return L + [""]
