"""Geometry regressions, plus an optional audit of the supplied textbook.

TFKP_TEST_PDF points to a private local PDF; the PDF is never bundled in tests.
"""
import os
import re

import pytest

from prepare_data import extract_pdf, repair_font_encoding
from rag.pdf_math import Atom, Rule, cmap_codes, collapse_pieces, glyph_label, linear, render_math


def stats():
    return {"roots": 0, "overlines": 0, "fractions": 0, "unassigned_rules": 0}


def a(text, x0, y0, x1, y1, baseline, size, number, kind="glyph"):
    return Atom(text, x0, y0, x1, y1, baseline, size, kind, frozenset({number}))


def compact(text):
    return re.sub(r"\s+", "", text)


def test_fraction_bar_is_not_overline_and_denominator_keeps_conjugation():
    atoms = [a("∂", 0, -1.4, 5, 5.9, 5.75, 10, 1), a("f", 6, -1.3, 10, 7.8, 5.75, 10, 2),
             a("∂", 0, 12.16, 5, 19.54, 19.32, 10, 3), a("z", 6.3, 14.9, 10.5, 19.43, 19.32, 10, 4)]
    report = stats()
    text, _ = render_math(atoms, [Rule(0, 12, 10, 1), Rule(6, 11, 13.6, 2)], report)
    assert compact(text) == r"\frac{∂f}{∂\overline{z}}"
    assert report["fractions"] == report["overlines"] == 1
    assert report["unassigned_rules"] == 0


def test_overline_covers_complete_exponential_not_just_base():
    atoms = [a("e", 0, 5.6, 4, 10.1, 10, 10, 1), a("z", 5, 2.8, 8, 6, 5.9, 7, 2)]
    text, _ = render_math(atoms, [Rule(0, 9, 1.5, 1)], stats())
    assert compact(text) == r"\overline{e^{z}}"


def test_nested_indices_stay_inside_exponent():
    atoms = [a("e", 0, 10.6, 4, 15.1, 15, 10, 1),
             a("z", 5, 7.8, 8, 11, 10.9, 7, 2), a("1", 8.5, 9, 10, 12.1, 12, 5, 3),
             a("+", 11, 7.8, 15, 11, 10.9, 7, 4), a("z", 16, 7.8, 19, 11, 10.9, 7, 5),
             a("2", 19.5, 9, 21, 12.1, 12, 5, 6)]
    assert compact(linear(atoms)) == "e^{z_{1}+z_{2}}"


def test_stretched_modulus_is_one_delimiter_with_all_source_ids():
    pieces = [a("|", 0, i * 6, 1, i * 6 + 6, i * 6 + 3, 10, i, "piece") for i in range(4)]
    merged = collapse_pieces(pieces)
    assert len(merged) == 1 and merged[0].text == "|"
    assert merged[0].ids == frozenset(range(4))


@pytest.mark.parametrize("name,raw,expected", [
    ("summationdisplay", "X", "∑"), ("integraltext", "R", "∫"),
    ("parenleftbigg", "\x12", "("), ("vextendsingle", "\x0c", "|"),
    ("slashbigg", "\x1e", "/"),
])
def test_cmex_symbols_are_decoded_by_font_glyph_names(name, raw, expected):
    assert glyph_label(name, raw, "CMEX10")[0] == expected


def test_cmap_preserves_source_encoding_for_unicode_glyphs():
    result = cmap_codes("1 beginbfchar <40> <2202> endbfchar 1 beginbfrange <0B> <0D> <03B1> endbfrange")
    assert result == {"∂": 64, "α": 11, "β": 12, "γ": 13}


def test_legacy_text_accent_is_preserved_as_combining_unicode():
    assert repair_font_encoding("Ðóø\x01å") == "Руше\u0301"


def test_builtin_prose_font_is_not_treated_as_math_or_cp1251(tmp_path):
    import pymupdf
    pdf = tmp_path / "sample.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 70), "Plain digital PDF: cafe and cafe.")
        document.save(pdf)
    blocks, metadata = extract_pdf(pdf, tmp_path / "audit")
    assert len(blocks) == 1
    assert blocks[0].text == "Plain digital PDF: cafe and cafe."
    assert metadata["pdf_pages"] == 1
    assert (tmp_path / "audit/page_1.txt").is_file()


def test_supplied_book_all_pages_and_audited_formulas(tmp_path):
    pdf = os.environ.get("TFKP_TEST_PDF")
    if not pdf:
        pytest.skip("Set TFKP_TEST_PDF to audit the private supplied textbook")
    blocks, metadata = extract_pdf(pdf, tmp_path / "audit")
    second, again = extract_pdf(pdf)
    assert [b.model_dump() for b in blocks] == [b.model_dump() for b in second]
    assert metadata == again
    assert len(blocks) == metadata["pdf_pages"] == 147
    assert not metadata["warnings"]
    assert len({b.block_id for b in blocks}) == len(blocks)
    for b in blocks:
        assert not any(ord(c) < 32 and c not in "\n\t\r" for c in b.text)
        assert r"\glyph" not in b.text
    pages = {b.page: compact(b.text) for b in blocks}
    expected = {
        11: [r"\frac{∂u}{∂x}=\frac{∂v}{∂y}", r"\frac{∂u}{∂y}=−\frac{∂v}{∂x}"],
        12: [r"\frac{∂f}{∂\overline{z}}=0", r"d\overline{z}=dx−idy",
             r"\frac{∂f}{∂z}=\frac{1}{2}(\frac{∂f}{∂x}−i\frac{∂f}{∂y})"],
        13: [r"\frac{∂^{2}u}{∂x^{2}}+\frac{∂^{2}u}{∂y^{2}}=0"],
        20: [r"\overline{e^{z}}=e^{\overline{z}}", r"\overline{e^{iy}}=e^{−iy}",
             r"\sqrt[n]{n!}", r"e^{z_{1}+z_{2}}=e^{z_{1}}e^{z_{2}}", "∑_{n=0}^{∞}"],
        21: [r"\frac{1}{2i}(e^{iz}−e^{−iz})"],
        31: [r"\frac{1}{2πi}∫_{γ}\frac{φ(ζ)}{ζ−z}dζ"],
        33: [r"F^{(n)}(z)=\frac{n!}{2πi}∫_{γ}\frac{φ(ζ)}{(ζ−z)^{n+1}}dζ"],
        34: [r"f^{(n)}(z)=\frac{n!}{2πi}∫_{γ_{r}}\frac{f(ζ)dζ}{(ζ−z)^{n+1}}"],
        36: [r"\overline{\mathcal{O}_{r}(z_{0})}⊂D", r"c_{n}=\frac{f^{(n)}(z_{0})}{n!}",
             r"|\frac{z−z_{0}}{ζ−z_{0}}|=\frac{|z−z_{0}|}{r}<1",
             r"\frac{1}{(ζ−z_{0})(1−\frac{z−z_{0}}{ζ−z_{0}})}"],
        37: ["невыше,чем", "тождественнопостоянна"],
        38: [r"\dot{\mathcal{O}}_{δ}(a)", r"\begin{cases}e^{−1/x^{2}}", "принадлежащуюобласти"],
    }
    for page, fragments in expected.items():
        for fragment in fragments:
            assert compact(fragment) in pages[page], f"page {page}: missing {fragment}"
