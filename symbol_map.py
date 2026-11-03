"""
symbol_map.py

Turns a picture of what the student has written into a list of NAMED, touchable symbols.
Ported from 0_test_sam.py (same SAM 3.1 idea), but async and tuned for the student's own drawing.

Pipeline:
  1. SAM 3.1 is asked for each kind of symbol ("equals sign", "digit", "letter", ...); answers are merged.
  2. An "equals sign" is always ONE item (two stacked bars are merged). A lone bar stays a lone bar.
  3. Ink that SAM missed is added from the pixels, so no symbol is ever lost.
  4. Muse Spark looks at each symbol and says what is ACTUALLY drawn (it does not guess what the student
     meant), so a half-finished "=" is spoken as "a single horizontal line".

Boxes are returned as FRACTIONS (0-1) of the picture, so the browser can scale them to any canvas size.
"""
import os
import re
import io
import json
import base64
import asyncio

import numpy as np
import cv2
import httpx
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

SAM_ENDPOINT = "https://api.meta.ai/v1/responses"
CHAT_ENDPOINT = "https://api.meta.ai/v1/chat/completions"
MAX_SIDE = 1280          # the picture is downscaled to this before going to Meta's models
SAM_CONCURRENCY = 6
MAX_SYMBOLS = 30

# Most specific first: if two prompts find the same symbol, the earlier one wins and names it.
SYMBOL_PROMPTS = [
    "equals sign", "plus sign", "minus sign", "fraction line", "integral sign",
    "parentheses", "digit", "letter", "math symbol", "handwritten character",
]
GENERIC_PROMPTS = {"math symbol", "handwritten character"}

BOX_RE = re.compile(r"<\|box;x1=(\d+(?:\.\d+)?);y1=(\d+(?:\.\d+)?);x2=(\d+(?:\.\d+)?);y2=(\d+(?:\.\d+)?);")


def _key():
    return os.getenv("META_API_KEY")


# ----------------------------------------------------------------------------- SAM call
def _collect_texts(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _collect_texts(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _collect_texts(v)]
    return []


async def _call_sam(client, sem, data_url, prompt):
    payload = {
        "model": "sam-3.1",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    async with sem:
        try:
            r = await client.post(SAM_ENDPOINT, headers=headers, json=payload, timeout=30.0)
        except Exception as e:
            print(f"   SAM '{prompt}' failed: {e}")
            return []
    if r.status_code != 200:
        print(f"   SAM '{prompt}': HTTP {r.status_code}: {r.text[:200]}")
        return []
    try:
        text = "\n".join(_collect_texts(r.json()))
    except ValueError:
        return []
    boxes = [[int(float(v)) for v in m] for m in BOX_RE.findall(text)]
    print(f"   SAM '{prompt}': {len(boxes)} box(es)")
    return boxes


# ----------------------------------------------------------------------------- geometry
def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u else 0.0


def _inside_frac(small, big):
    return _inter(small, big) / _area(small) if _area(small) else 0.0


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


# ----------------------------------------------------------------------------- equals sign = ONE item
def _consolidate_equals(items):
    # (a) anything mostly inside a SAM "equals sign" box belongs to it (e.g. each bar found as a "minus")
    eq_items = [it for it in items if it["label"] == "equals sign"]
    if eq_items:
        items = [it for it in items
                 if it["label"] == "equals sign"
                 or not any(_inside_frac(it["box"], e["box"]) >= 0.6 for e in eq_items)]

    # (b) safety net: two stacked flat bars of similar width are ONE equals sign
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i]["box"], items[j]["box"]
                aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
                if min(aw, bw, ah, bh) < 1:
                    continue
                flat = aw >= 2.0 * ah and bw >= 2.0 * bh
                similar = 0.6 <= aw / bw <= 1.67
                aligned = (min(a[2], b[2]) - max(a[0], b[0])) >= 0.7 * min(aw, bw)
                gap = max(a[1], b[1]) - min(a[3], b[3])
                if flat and similar and aligned and gap <= 0.6 * max(aw, bw):
                    items[i] = {"box": _union(a, b), "label": "equals sign", "source": "merged"}
                    del items[j]
                    merged = True
                    break
            if merged:
                break
    return items


# ----------------------------------------------------------------------------- ink check (what SAM missed)
def _ink_components(ink, region):
    x1, y1, x2, y2 = region
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink[y1:y2, x1:x2], connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 8:
            continue
        comps.append({"box": [x1 + int(x), y1 + int(y), x1 + int(x + w), y1 + int(y + h)],
                      "mask": labels == i, "ink": int(a)})
    return comps


def _coverage(items, comps, region):
    x1, y1, x2, y2 = region
    covered = np.zeros((y2 - y1, x2 - x1), bool)
    for it in items:
        bx1, by1, bx2, by2 = it["box"]
        covered[max(0, by1 - y1):max(0, by2 - y1), max(0, bx1 - x1):max(0, bx2 - x1)] = True
    return [float(covered[c["mask"]].mean()) for c in comps]


# ----------------------------------------------------------------------------- naming each symbol (Muse Spark)
def _crop_b64(img, box):
    pad = 8
    W, H = img.size
    crop = img.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)))
    longest = max(crop.size)
    if longest < 96:
        s = 96 / longest
        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    crop = ImageOps.expand(crop, border=12, fill="white")
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fallback_name(item):
    return "something written here" if item["label"] == "symbol" else item["label"]


NAME_PROMPT = (
    "You help a blind student explore their own handwriting by touch. The picture shows ONE handwritten "
    "math symbol. Say what is ACTUALLY drawn, in at most 6 words, the way it would be spoken aloud, for example: "
    "'the letter y', 'the letter x', 'the digit 2', 'an equals sign', 'a plus sign', 'a minus sign', "
    "'a left parenthesis'. Handwritten letters and digits are far more common than special symbols: if the shape "
    "could be a letter or a digit, say the letter or digit. Only say 'an integral sign' if it is clearly a tall, "
    "stretched S shape. Do NOT guess what the student meant to write. If a symbol is incomplete, describe what "
    "is there: a single horizontal bar is 'a single horizontal line', NOT an equals sign. If it is just a mess "
    "of strokes, say 'a scribble'. "
    "Output ONLY the spoken name, nothing else."
)


def _name_prompt(hint):
    # The equation on the board tells us which look-alike letters and digits are likely (y vs a long S, etc.).
    # It is only a tie-breaker: what is really drawn always wins.
    if not hint:
        return NAME_PROMPT
    return (NAME_PROMPT + f" The student is copying an equation that reads: '{hint}'. Use that ONLY to choose "
            "between look-alike letters and digits. If the drawing is clearly something else, say what it really is.")


async def _name_one(client, sem, img, item, fallback, hint=""):
    payload = {
        "model": "muse-spark-1.3",
        "messages": [
            {"role": "system", "content": _name_prompt(hint)},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _crop_b64(img, item["box"])}}]},
        ],
    }
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    for attempt in (1, 2):
        try:
            async with sem:
                r = await client.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=20.0)
            if r.status_code != 200:
                print(f"   Spark naming: HTTP {r.status_code}: {r.text[:200]}")
                continue
            text = r.json()["choices"][0]["message"]["content"].strip().strip("\"'`.").strip()
            text = text.split("\n")[0][:60]
            if text:
                return text
        except Exception as e:
            print(f"   Spark naming attempt {attempt} failed: {type(e).__name__}: {e!r}")
    return fallback


async def _name_symbols(client, img, items, hint=""):
    # One small request per symbol, all in parallel: far more reliable than one big multi-image request.
    names = [_fallback_name(it) for it in items]
    if not items or not _key():
        return names
    sem = asyncio.Semaphore(SAM_CONCURRENCY)
    return list(await asyncio.gather(*(_name_one(client, sem, img, it, names[i], hint) for i, it in enumerate(items))))



# ----------------------------------------------------------------------------- main entry point
def _prepare_image(image_b64):
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))     # flatten any transparency onto white
    img = Image.alpha_composite(bg, img).convert("RGB")
    longest = max(img.size)
    if longest > MAX_SIDE:
        s = MAX_SIDE / longest
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    return img


async def build_symbol_map(image_b64: str, hint: str = ""):
    """Returns [{"id", "label", "box": [fx1, fy1, fx2, fy2], "source"}] sorted left to right."""
    try:
        img = _prepare_image(image_b64)
    except Exception as e:
        print(f"Symbol map: could not read the picture: {e}")
        return []

    W, H = img.size
    gray = np.array(img.convert("L"))
    ink = (gray < 128).astype(np.uint8) * 255
    ys, xs = np.where(ink > 0)
    if len(xs) < 30:
        return []                                              # nothing written yet

    pad = 6
    region = [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
              min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)]
    comps = _ink_components(ink, region)
    if not comps:
        return []

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    async with httpx.AsyncClient() as client:
        # 1. SAM, one call per symbol kind (in parallel)
        if _key():
            sem = asyncio.Semaphore(SAM_CONCURRENCY)
            sam_results = await asyncio.gather(*(_call_sam(client, sem, data_url, p) for p in SYMBOL_PROMPTS))
        else:
            print("Symbol map: META_API_KEY missing - using ink pixels only")
            sam_results = [[] for _ in SYMBOL_PROMPTS]

        items = []
        for prompt, boxes in zip(SYMBOL_PROMPTS, sam_results):
            for b in boxes:
                b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
                if _area(b) < 20:
                    continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                    continue
                if any(_iou(b, it["box"]) > 0.5 or _inside_frac(b, it["box"]) > 0.8 for it in items):
                    continue                                   # duplicate / part of something already found
                if sum(1 for c in comps if _inside_frac(c["box"], b) > 0.8) >= 3:
                    continue                                   # a group box, not a single symbol
                items.append({"box": b, "label": "symbol" if prompt in GENERIC_PROMPTS else prompt, "source": "sam"})

        items = _consolidate_equals(items)

        # 2. add any ink SAM missed
        per_comp = _coverage(items, comps, region)
        for c, f in zip(comps, per_comp):
            if f < 0.5:
                items.append({"box": c["box"], "label": "symbol", "source": "ink"})
        items = _consolidate_equals(items)

        items.sort(key=lambda it: ((it["box"][0] + it["box"][2]) / 2, it["box"][1]))
        items = items[:MAX_SYMBOLS]

        # 3. say what each one actually is
        names = await _name_symbols(client, img, items, hint)

    segments = []
    for i, (it, name) in enumerate(zip(items, names), 1):
        b = it["box"]
        segments.append({"id": i, "label": name, "sam_label": it["label"], "source": it["source"],
                         "box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H]})
    print(f"Symbol map: {len(segments)} symbols -> " + "; ".join(s["label"] for s in segments))
    return segments