"""轻量评分原语：被各 judge 复用。"""
from __future__ import annotations
import re
import string


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # 去掉 markdown 强调符号
    s = re.sub(r"[*_`]", "", s)
    # 去首尾标点（含 {} () [] 这种包装；模型常输出 {1}/(B) 之类）
    s = s.strip(string.punctuation + " ")
    # 去冠词/无意义前缀，但只在前缀后面还有内容时才剥（避免吞掉孤立的 'A' 选项）
    s = re.sub(r"^(answer|the answer is|final answer)[\s:：]*", "", s).strip()
    s = re.sub(r"^(a|an|the)\s+", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def text_exact_match(pred: str, gt: str, aliases: list[str] | None = None) -> bool:
    p = normalize_text(pred)
    if not p:
        return False
    cand = [gt] + list(aliases or [])
    return any(p == normalize_text(c) for c in cand if c)


_LETTER_PATS = [
    r"\\boxed\{\s*([A-J])\s*\}",
    r"<answer>\s*([A-J])\s*</answer>",
    r"\b(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-J])\b",
    r"^\s*\(?([A-J])\)?\s*[\.\)：:]",
    r"\b([A-J])\s*$",
]


def extract_choice_letter(text: str, valid: list[str] | None = None) -> str | None:
    if not text:
        return None
    valid = [v.upper() for v in (valid or list("ABCDEFGHIJ"))]
    upper = text
    for pat in _LETTER_PATS:
        m = re.search(pat, upper, re.IGNORECASE | re.MULTILINE)
        if m:
            ch = m.group(1).upper()
            if ch in valid:
                return ch
    # 最后兜底：找最后一个出现的合法字母
    last = None
    for m in re.finditer(r"\b([A-J])\b", upper):
        if m.group(1).upper() in valid:
            last = m.group(1).upper()
    return last


def extract_yes_no(text: str) -> int | None:
    """返回 1=yes, 0=no, None=抽不出。"""
    if not text:
        return None
    low = text.lower()
    # 先看明显否定
    yes_hits = bool(re.search(r"\b(yes|true|correct|是的|对|正确)\b", low))
    no_hits = bool(re.search(r"\b(no|false|incorrect|wrong|不是|否|错误)\b", low))
    if yes_hits and not no_hits:
        return 1
    if no_hits and not yes_hits:
        return 0
    # 同时出现：取最后一次出现的
    pos_yes = max([m.start() for m in re.finditer(r"\byes\b|\b正确\b|\b对\b", low)], default=-1)
    pos_no = max([m.start() for m in re.finditer(r"\bno\b|\b不对\b|\b错误\b", low)], default=-1)
    if pos_yes == -1 and pos_no == -1:
        return None
    return 1 if pos_yes > pos_no else 0


_NUM_RE = re.compile(r"-?\d+(?:[\.,]\d+)?(?:[eE][+-]?\d+)?")


def extract_number(text: str) -> float | None:
    if not text:
        return None
    # 优先 \boxed
    m = re.search(r"\\boxed\{\s*([-+]?\d[^}]*)\}", text)
    cand_text = m.group(1) if m else text
    nums = _NUM_RE.findall(cand_text.replace(",", ""))
    if not nums:
        nums = _NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def numeric_close(pred: float, gt: float, rel_tol: float = 0.05) -> bool:
    if gt == 0:
        return abs(pred) < 1e-6
    return abs(pred - gt) / abs(gt) <= rel_tol


def anls(pred: str, gt: str, tau: float = 0.5) -> float:
    """官方 ANLS：1 - normalized levenshtein；< tau 算 0。"""
    p = (pred or "").strip().lower()
    g = (gt or "").strip().lower()
    if not p and not g:
        return 1.0
    nl = _levenshtein(p, g) / max(len(p), len(g), 1)
    s = 1.0 - nl
    return s if s >= tau else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[-1]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


_BBOX_RE_LIST = [
    re.compile(r"\[\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\]"),
    re.compile(r"<box>\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*</box>"),
    re.compile(r"\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)"),
]


def extract_bbox(text: str) -> list[float] | None:
    """从模型输出抽 [x1,y1,x2,y2]。坐标空间不在这里转换。"""
    if not text:
        return None
    for pat in _BBOX_RE_LIST:
        m = pat.search(text)
        if m:
            try:
                return [float(m.group(i)) for i in range(1, 5)]
            except ValueError:
                continue
    return None
