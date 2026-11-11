"""Deterministic transcription of digital TeX PDFs using fonts and page geometry.

No OCR, language model, question IDs, page-specific formulas, or QA data is used.
The result is a geometric transcription, not a mathematical correctness proof.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from statistics import median


@dataclass
class Atom:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    kind: str = "glyph"
    ids: frozenset[int] = frozenset()
    source_x: float | None = None


@dataclass(frozen=True)
class Rule:
    x0: float
    x1: float
    y: float
    id: int


def combined(text, items, baseline=None, kind="group"):
    largest = max(a.size for a in items)
    return Atom(text, min(a.x0 for a in items), min(a.y0 for a in items),
                max(a.x1 for a in items), max(a.y1 for a in items),
                baseline if baseline is not None else median(a.baseline for a in items if a.size >= largest * .9),
                largest, kind, frozenset().union(*(a.ids for a in items)),
                min((a.source_x for a in items if a.source_x is not None), default=None))


def cmap_codes(stream):
    """Reverse a PDF ToUnicode CMap; supports the bfchar/bfrange forms in TeX PDFs."""
    result = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", stream, re.S):
        for source, target in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", block):
            result.setdefault(bytes.fromhex(target).decode("utf-16-be"), int(source, 16))
    for block in re.findall(r"beginbfrange(.*?)endbfrange", stream, re.S):
        for lo, hi, start in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", block):
            lo, hi, start = int(lo, 16), int(hi, 16), int(start, 16)
            if hi - lo > 65536:
                raise ValueError("Unreasonably large font CMap range")
            for code in range(lo, hi + 1):
                result.setdefault(chr(start + code - lo), code)
    return result


def glyph_label(name, original, font):
    # These are font glyph names, not text substitutions based on mathematical context.
    operators = {"summation": "∑", "integral": "∫", "product": "∏", "union": "⋃", "intersection": "⋂"}
    for prefix, label in operators.items():
        if name.startswith(prefix) and "CMEX" in font:
            return label, "operator_inline" if name.endswith("text") else "operator"
    for prefix, label in [("parenleft", "("), ("parenright", ")"),
                          ("bracketleft", "["), ("bracketright", "]"),
                          ("braceleft", "\\{"), ("braceright", "\\}"),
                          ("angbracketleft", "⟨"), ("angbracketright", "⟩")]:
        if name.startswith(prefix):
            return label, "piece" if name.endswith(("tp", "bt", "mid", "ex")) else "delimiter"
    if name.startswith("vextendsingle"):
        return "|", "piece"
    if name.startswith("vextenddouble"):
        return "‖", "piece"
    if name == "braceex":
        return "", "extension"
    if name.startswith("slash"):
        return "/", "delimiter"
    if name.startswith("radical"):
        return "√", "radical"
    if name in {"dotaccent", "dieresis", "hat", "tilde", "acute", "grave", "breve", "check", "ring", "vecaccent", "macron"}:
        return {"dotaccent": "\\dot", "dieresis": "\\ddot", "vecaccent": "\\vec", "macron": "\\bar"}.get(name, "\\" + name), "accent"
    if name == "backslash":
        return "\\setminus ", "glyph"
    if "CMSY" in font and len(name) == 1 and name.isupper():
        return "\\mathcal{" + name + "}", "glyph"
    if "MSBM" in font and len(name) == 1 and name.isupper():
        return "\\mathbb{" + name + "}", "glyph"
    if original in {"\uf8eb", "\uf8ec", "\uf8ed"}:
        return "(", "piece"
    if original in {"\uf8f6", "\uf8f7", "\uf8f8"}:
        return ")", "piece"
    return original, "glyph"


class Font:
    def __init__(self, document, xref, kind, name):
        self.kind, self.name = kind, name
        self.is_math = kind == "Type1" and bool(re.match(r"^(CM|MS|EU|LM)", name))
        self.glyphs, self.encoding, self.codes, self.bounds = None, [], {}, {}
        if self.is_math:
            from fontTools.t1Lib import T1Font
            # Read the embedded Type1 font's encoding and outlines, without OCR/model calls.
            font = T1Font.__new__(T1Font)
            font.data = document.extract_font(xref)[3]
            font.encoding = "ascii"
            font.parse()
            self.glyphs = font.getGlyphSet()
            self.encoding = font.font["Encoding"]
            match = re.search(r"/ToUnicode\s+(\d+)\s+0\s+R", document.xref_object(xref))
            if match:
                self.codes = cmap_codes(document.xref_stream(int(match[1])).decode("latin1"))

    def atom(self, char, size, index, warnings):
        original = char["c"]
        code = self.codes.get(original, ord(original) if len(original) == 1 else -1)
        name = self.encoding[code] if 0 <= code < len(self.encoding) else ""
        text, kind = glyph_label(name, original, self.name)
        x, y = char["origin"]
        box = char["bbox"]
        if self.glyphs is not None and name in self.glyphs:
            if name not in self.bounds:
                from fontTools.pens.boundsPen import BoundsPen
                pen = BoundsPen(self.glyphs)
                self.glyphs[name].draw(pen)
                self.bounds[name] = pen.bounds
            bounds = self.bounds[name]
            if bounds:
                a, b, c, d = bounds
                box = (x + a * size / 1000, y - d * size / 1000,
                       x + c * size / 1000, y - b * size / 1000)
        if any(ord(c) < 32 or 0xE000 <= ord(c) <= 0xF8FF for c in text):
            warnings.append({"type": "unknown_math_glyph", "font": self.name,
                             "glyph": name, "codepoints": [ord(c) for c in text]})
            text = "\\glyph{" + self.name + "}{" + str(code) + "}"
        if "CMEX" in self.name and kind in {"operator", "operator_inline", "delimiter", "piece", "extension"}:
            y = (box[1] + box[3]) / 2 + .25 * size
        return Atom(text, *box, y, size, kind, frozenset({index}), x)


def horizontal_rules(page):
    result = []
    for drawing in page.get_drawings():
        if (drawing.get("width") or 0) > 1.5:
            continue
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:]
                if abs(a.y - b.y) < .15 and abs(a.x - b.x) > .4:
                    result.append(Rule(min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2, len(result)))
            elif item[0] == "re":
                rect = item[1]
                if rect.height <= 1 and rect.width > 1:
                    result.append(Rule(rect.x0, rect.x1, (rect.y0 + rect.y1) / 2, len(result)))
    return result


def collapse_pieces(atoms):
    remaining, result = list(atoms), []
    while remaining:
        atom = remaining.pop(0)
        if atom.kind not in {"piece", "extension"}:
            result.append(atom)
            continue
        group = [atom]
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                ax = atom.source_x if atom.source_x is not None else atom.x0
                bx = other.source_x if other.source_x is not None else other.x0
                if (other.kind in {"piece", "extension"} and (not other.text or not atom.text or other.text == atom.text)
                        and abs(bx - ax) < 1
                        and min(min(abs(other.y0 - a.y1), abs(other.y1 - a.y0)) for a in group) < .2 * atom.size):
                    group.append(other)
                    remaining.remove(other)
                    changed = True
        text = next((a.text for a in group if a.text), "\\glyph{brace-extension}")
        merged = combined(text, group, kind="delimiter")
        merged.baseline = (merged.y0 + merged.y1) / 2 + .25 * merged.size
        result.append(merged)
    return result


def inline(atoms, baseline):
    result = []
    atoms = sorted(atoms, key=lambda a: (a.x0, a.y0))
    i = 0
    while i < len(atoms):
        a = atoms[i]
        # Only smaller glyphs are scripts: full-sized fraction nodes are body atoms.
        if result and a.size < .88 * result[-1].size and abs(a.baseline - baseline) > .7:
            owner = result[-1]
            if a.x0 <= owner.x1 + owner.size * .9:
                group = [a]
                j = i + 1
                while j < len(atoms):
                    b = atoms[j]
                    if (b.size < .88 * owner.size and abs(b.baseline - a.baseline) < .75 * owner.size
                            and (b.baseline < baseline - .7) == (a.baseline < baseline - .7)
                            and b.x0 - group[-1].x1 < .7 * owner.size):
                        group.append(b)
                        j += 1
                    else:
                        break
                marker = "^" if a.baseline < baseline else "_"
                result[-1] = replace(combined(owner.text + marker + "{" + linear(group) + "}",
                                             [owner, *group], baseline=baseline), size=owner.size)
                i = j
                continue
        result.append(a)
        i += 1
    text = ""
    previous = None
    for atom in result:
        if atom.kind == "case_text":
            text += " & "
        if previous and atom.x0 - previous.x1 > .25 * max(atom.size, previous.size):
            text += " "
        text += atom.text
        previous = atom
    return text


def linear(atoms):
    if not atoms:
        return ""
    cases = [a for a in atoms if a.text == "\\{" and a.y1 - a.y0 > 1.6 * a.size
             and not any(b.text == "\\}" and b.x0 > a.x0 and abs(b.y0 - a.y0) < a.size for b in atoms)]
    if cases:
        brace = min(cases, key=lambda a: a.x0)
        head = [a for a in atoms if a is not brace and a.x1 <= brace.x0 + .5]
        body = [a for a in atoms if a is not brace and a not in head]
        return linear(head) + " \\begin{cases}" + linear(body) + "\\end{cases}"
    size = max(a.size for a in atoms)
    anchors = sorted(a.baseline for a in atoms if a.size >= .9 * size)
    rows = []
    for y in anchors:
        if not rows or abs(y - rows[-1]) > 1.0:
            rows.append(y)
    grouped = [[] for _ in rows]
    for atom in atoms:
        row = min(range(len(rows)), key=lambda i: abs(rows[i] - atom.baseline))
        grouped[row].append(atom)
    return " \\\\\n".join(inline(group, y) for group, y in zip(grouped, rows))


def render_math(raw_atoms, rules, stats):
    atoms = collapse_pieces(raw_atoms)
    used = set()
    for accent in [a for a in atoms if a.kind == "accent"]:
        targets = [a for a in atoms if a is not accent and a.kind not in {"accent", "delimiter"}
                   and accent.baseline < a.baseline < accent.baseline + 1.5 * a.size
                   and abs((a.x0 + a.x1 - accent.x0 - accent.x1) / 2) < .5 * a.size]
        if targets:
            target = min(targets, key=lambda a: abs((a.x0 + a.x1 - accent.x0 - accent.x1) / 2))
            node = combined(accent.text + "{" + target.text + "}", [accent, target], baseline=target.baseline)
            atoms = [a for a in atoms if a not in [accent, target]] + [node]
    # A radical's horizontal roof belongs to the root, not to an overline.
    for root in [a for a in atoms if a.kind == "radical"]:
        roofs = [r for r in rules if abs(r.y - root.y0) < 2 and root.x0 < r.x0 <= root.x1 + 2]
        if not roofs:
            continue
        roof = min(roofs, key=lambda r: abs(r.y - root.y0))
        content = [a for a in atoms if a is not root and roof.x0 - .5 <= (a.x0 + a.x1) / 2 <= roof.x1 + .5
                   and roof.y <= a.y0 and a.baseline - roof.y < 2 * root.size]
        if not content:
            continue
        content_baseline = median(a.baseline for a in content if a.size >= .9 * max(c.size for c in content))
        index = [a for a in atoms if a is not root and a not in content and a.size < .8 * root.size
                 and root.x0 - .5 * root.size < a.x0 < root.x1 and a.baseline < content_baseline - 1]
        node = combined("\\sqrt" + ("[" + linear(index) + "]" if index else "") + "{" + linear(content) + "}",
                        [root, *content, *index], baseline=content_baseline)
        atoms = [a for a in atoms if a not in [root, *content, *index]] + [node]
        used.add(roof.id)
        stats["roots"] += 1
    # TeX overlines sit one rule-gap above the highest ink, including raised exponents.
    for rule in sorted(rules, key=lambda r: r.x1 - r.x0):
        if rule.id in used:
            continue
        # A visible numerator immediately above the line makes it a fraction.
        # A second rule between that numerator and this line blocks the relation:
        # this distinguishes a bar over a denominator's z from the fraction above it.
        visible_above = [a for a in atoms if rule.x0 - .65 <= (a.x0 + a.x1) / 2 <= rule.x1 + .65
                         and 0 <= rule.y - a.y1 < .65 * a.size
                         and not any(a.y1 + .1 < r.y < rule.y - .1
                                     and r.x0 <= (a.x0 + a.x1) / 2 <= r.x1 for r in rules)]
        if visible_above:
            continue
        below = [a for a in atoms if rule.x0 - .7 <= (a.x0 + a.x1) / 2 <= rule.x1 + .7
                 and .04 * a.size <= a.y0 - rule.y <= .17 * max(n.size for n in atoms)]
        if not below:
            continue
        top = min(a.y0 for a in below)
        size = max(a.size for a in below)
        covered = [a for a in atoms if rule.x0 - .7 <= (a.x0 + a.x1) / 2 <= rule.x1 + .7
                   and a.y0 >= rule.y - .1 and a.y1 <= top + 1.8 * size]
        if not covered:
            continue
        node = combined("\\overline{" + linear(covered) + "}", covered)
        node.y0 = rule.y
        atoms = [a for a in atoms if a not in covered] + [node]
        used.add(rule.id)
        stats["overlines"] += 1
    # Short bars first allow nested fractions to become atoms of outer fractions.
    for rule in sorted(rules, key=lambda r: r.x1 - r.x0):
        if rule.id in used:
            continue
        inside = [a for a in atoms if rule.x0 - .65 <= (a.x0 + a.x1) / 2 <= rule.x1 + .65]
        size = max((a.size for a in inside), default=0)
        above = [a for a in inside if 0 < rule.y - a.baseline < 2.2 * size]
        below = [a for a in inside if 0 < a.baseline - rule.y < 2.2 * size]
        if not above or not below:
            continue
        ay, by = max(a.baseline for a in above), min(a.baseline for a in below)
        numerator = [a for a in above if ay - a.baseline <= .65 * max(a.size for a in above)]
        denominator = [a for a in below if a.baseline - by <= .65 * max(a.size for a in below)]
        parts = numerator + denominator
        node = combined("\\frac{" + linear(numerator) + "}{" + linear(denominator) + "}", parts,
                        baseline=rule.y + .25 * max(a.size for a in parts))
        atoms = [a for a in atoms if a not in parts] + [node]
        used.add(rule.id)
        stats["fractions"] += 1
    for operator in [a for a in atoms if a.kind in {"operator", "operator_inline"}]:
        supers, subs = [], []
        margin = (1.6 if operator.kind == "operator_inline" else .6) * operator.size
        for a in atoms:
            if (a is not operator and a.size < .85 * operator.size
                    and operator.x0 - .6 * operator.size <= (a.x0 + a.x1) / 2 <= operator.x1 + margin
                    and abs(a.baseline - operator.baseline) < 2.5 * operator.size):
                (supers if a.baseline < operator.baseline else subs).append(a)
        text = operator.text
        if subs:
            text += "_{" + linear(subs) + "}"
        if supers:
            text += "^{" + linear(supers) + "}"
        node = combined(text, [operator, *supers, *subs], baseline=operator.baseline, kind="complete_operator")
        atoms = [a for a in atoms if a not in [operator, *supers, *subs]] + [node]
    # Every source glyph must survive the transformations exactly once.
    expected = frozenset().union(*(a.ids for a in raw_atoms))
    actual = [i for a in atoms for i in a.ids]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("PDF math transcription lost or duplicated source glyphs")
    stats["unassigned_rules"] += len(rules) - len(used)
    stats.setdefault("unassigned_rule_positions", []).extend(
        {"x0": round(r.x0, 3), "x1": round(r.x1, 3), "y": round(r.y, 3)} for r in rules if r.id not in used
    )
    return linear(atoms), atoms


class PDFMathExtractor:
    def __init__(self, document, repair_text):
        self.document, self.repair_text = document, repair_text
        self.fonts = {}

    def page(self, page):
        fonts = {}
        for xref, _, kind, name, *_ in page.get_fonts():
            if xref not in self.fonts:
                self.fonts[xref] = Font(self.document, xref, kind, name.split("+")[-1])
            fonts[name.split("+")[-1]] = self.fonts[xref]
        rules = horizontal_rules(page)
        stats = {"overlines": 0, "fractions": 0, "roots": 0, "unassigned_rules": 0, "math_runs": 0}
        stats["vector_figure_paths"] = sum(any(item[0] == "c" for item in d["items"]) for d in page.get_drawings())
        stats["image_blocks"] = sum(b.get("type") == 1 for b in page.get_text("dict")["blocks"])
        warnings, parts, pending = [], [], []
        cursor_y, cursor_x, cursor_size = None, 0, 10
        source_glyphs = 0
        events, math_atoms = [], []
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = fonts.get(span["font"])
                    if font is None:
                        raise ValueError(f"Unknown PDF font {span['font']}")
                    atoms = []
                    if font.is_math:
                        for char in span["chars"]:
                            if char["c"] != " ":
                                source_glyphs += 1
                                atoms.append(font.atom(char, span["size"], source_glyphs, warnings))
                        math_atoms.extend(atoms)
                    events.append((font, span, atoms))
        collapsed = collapse_pieces(math_atoms)
        case_regions = [a for a in collapsed if a.text == "\\{" and a.y1 - a.y0 > 1.6 * a.size
                        and not any(b.text == "\\}" and b.x0 > a.x0 and abs(b.y0 - a.y0) < a.size for b in collapsed)]

        def flush_math():
            nonlocal cursor_y, cursor_x, cursor_size
            if not pending:
                return
            xmin, xmax = min(a.x0 for a in pending), max(a.x1 for a in pending)
            ymin, ymax = min(a.y0 for a in pending), max(a.y1 for a in pending)
            local_rules = [r for r in rules if r.x0 >= xmin - 2 and r.x1 <= xmax + 2 and ymin - 4 <= r.y <= ymax]
            text, nodes = render_math(pending, local_rules, stats)
            y = min(a.baseline for a in nodes if a.size >= .9 * max(n.size for n in nodes))
            if cursor_y is not None and y - cursor_y > .8 * cursor_size:
                parts.append("\n")
            parts.append(" $" + text + "$ ")
            cursor_y = max(a.baseline for a in nodes if a.size >= .9 * max(n.size for n in nodes))
            cursor_x, cursor_size = max(a.x1 for a in nodes), max(a.size for a in nodes)
            stats["math_runs"] += 1
            pending.clear()

        for event_id, (font, span, atoms) in enumerate(events):
            if font.is_math:
                pending.extend(atoms)
                continue
            text = "".join(c["c"] for c in span["chars"])
            if font.kind == "Type3":
                text = self.repair_text(text)
            if not text.strip():
                continue
            if any(a.y0 <= span["origin"][1] <= a.y1 and span["bbox"][0] > a.x1 for a in case_regions):
                pending.append(Atom("\\text{" + text.strip() + "}", *span["bbox"],
                                    span["origin"][1], span["size"], "case_text", frozenset({-event_id - 1})))
                continue
            flush_math()
            y = span["origin"][1]
            if cursor_y is not None:
                gap = y - cursor_y
                if gap > .8 * span["size"]:
                    parts.append("\n\n" if gap > 1.8 * span["size"] else "\n")
                elif span["bbox"][0] - cursor_x > .12 * span["size"]:
                    parts.append(" ")
            parts.append(text)
            cursor_y, cursor_x, cursor_size = y, span["bbox"][2], span["size"]
        flush_math()
        text = unicodedata.normalize("NFC", "".join(parts))
        text = re.sub(r"([А-Яа-я])-\n([а-я])", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text).strip()
        return text, {**stats, "math_glyphs": source_glyphs, "warnings": warnings}
