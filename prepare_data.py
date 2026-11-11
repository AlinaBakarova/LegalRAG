"""Deterministic dataset extraction. QA answers are evaluation data, never index data."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from rag.io import sha256_file, write_json, write_jsonl
from rag.schemas import Block, Gold, Question


def repair_font_encoding(text):
    # This PDF's Cyrillic font maps CP1251 glyph codes to Latin-1 Unicode.
    # Preserve mathematical multiplication/division signs and Greek glyphs.
    def word(match):
        token = match.group()
        if not any(0xC0 <= ord(c) <= 0xFF and c not in "×÷" for c in token):
            return token
        return "".join(c.encode("latin1").decode("cp1251") if 0xC0 <= ord(c) <= 0xFF or c in "¨¸" else c for c in token)

    text = re.sub(r"[A-Za-zÀ-ÿ¨¸]+", word, text)
    # In the source's legacy Type3 text fonts byte 1 is a prefixed acute accent.
    text = re.sub("\x01([A-Za-zА-Яа-яЁё])", lambda m: m.group(1) + "\u0301", text)
    return text.replace("\x9f", "§").replace("\x15", "-").replace("\x16", "—")


def page_id(pdf_digest, page):
    return f"pdf_{pdf_digest[:12]}_p{page:04d}"


def extract_pdf(pdf_path, audit_dir=None):
    import pymupdf
    from fontTools import version as fonttools_version
    from rag.pdf_math import PDFMathExtractor

    audit = Path(audit_dir) if audit_dir else None
    if audit:
        if audit.exists() and any(audit.iterdir()):
            raise ValueError("Audit directory must be empty")
        audit.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(pdf_path)
    blocks, reports, section = [], [], "front matter"
    with pymupdf.open(pdf_path) as document:
        extractor = PDFMathExtractor(document, repair_font_encoding)
        count = len(document)
        for number, page in enumerate(document, 1):
            text, report = extractor.page(page)
            headings = re.findall(r"^\s*§\s*\d+\.\s*[^\n]+", text, re.M)
            if headings:
                section = headings[0].strip()
            if text:
                blocks.append(Block(block_id=page_id(digest, number), page=number, section=section, text=text))
            reports.append({"page": number, **report})
            if audit:
                (audit / f"page_{number}.txt").write_text(text + "\n", encoding="utf-8")
    review = [r["page"] for r in reports if r["warnings"] or r["unassigned_rules"] or r["vector_figure_paths"] or r["image_blocks"]]
    metadata = {"pdf_sha256": digest, "pdf_pages": count, "extractor_schema": "tex-pdf-geometry-v1",
                "extractor": f"PyMuPDF {pymupdf.VersionBind}; fonttools {fonttools_version}",
                "math_format": "Geometric TeX transcription with Unicode mathematical symbols",
                "page_reports": reports, "needs_manual_review_pages": review,
                "warnings": [{"page": r["page"], **w} for r in reports for w in r["warnings"]]}
    if audit:
        write_json(audit / "extraction_audit.json", metadata)
    return blocks, metadata


def parse_qa(markdown, pdf_digest):
    categories = {"условия": "conditions", "сопоставление": "comparison", "применение": "calculation", "недостаточность данных": "insufficient"}
    sections = list(re.finditer(r"^## (\d+)\. (.+)$", markdown, re.M))
    questions, golds = [], []
    for i, match in enumerate(sections):
        body = markdown[match.end():sections[i + 1].start() if i + 1 < len(sections) else len(markdown)]
        category = categories[re.search(r"\*\*Тип:\*\*\s*([^\.]+)\.", body).group(1).strip()]
        question = re.search(r"\*\*Вопрос\.\*\*\s*(.*?)(?=\n- \*\*)", body, re.S).group(1).strip()
        options = [{"id": m.group(1), "text": m.group(2).strip()} for m in re.finditer(
            r"^- \*\*([^.]+)\.\*\*\s*(.*?)(?=\n- \*\*|\n\*\*Ответ:)", body, re.M | re.S
        )]
        answer = re.search(r"\*\*Ответ:\*\*\s*([^\n]+)", body).group(1).strip().rstrip(".")
        abstain = answer == "Отказ"
        rationale = re.search(r"\*\*Ответ:\*\*[^\n]+\n+(.*?)(?=\n\*\*Источник:)", body, re.S).group(1).strip()
        page_text = re.search(r"\*\*Источник:\*\* PDF с\. ([\d, –-]+)\.", body).group(1)
        pages = [int(p) for p in re.findall(r"\d+", page_text)]
        qid = f"qa_{int(match.group(1)):03d}"
        questions.append(Question(id=qid, question=question, options=options))
        golds.append(Gold(
            id=qid, category=category, answer_ids=[] if abstain else [a.strip() for a in answer.split(",")],
            abstain=abstain, evidence_sets=[] if abstain else [[page_id(pdf_digest, page) for page in pages]],
            rationale=rationale, missing_information=rationale if abstain else "",
        ))
    if len(questions) != 15:
        raise ValueError(f"Expected the supplied 15 QA sections, got {len(questions)}")
    return questions, golds


def own_examples():
    # Authored evaluation questions. These constants are never imported by predict.py.
    return [
        ("conditions", "f голоморфна при 0<|z-a|<r и ограничена там. Какие выводы обязательны?", [
            "Точка a является устранимой особенностью.", "Точка a обязательно является полюсом.",
            "f можно единственным образом голоморфно продолжить в a.", "f обязана быть постоянной."], ["A", "C"], [50],
         "По теореме 8.3 ограниченность даёт устранимость. Доопределение пределом голоморфно; значение единственно по непрерывности.", ""),
        ("conditions", "Целая функция f не принимает два различных комплексных значения. Какие утверждения следуют из приведённой в учебнике формы теоремы Пикара?", [
            "f постоянна.", "f обязана тождественно равняться нулю.", "f не может быть целой трансцендентной функцией.", "У f обязательно есть полюс в конечной точке."], ["A", "C"], [53],
         "На с. 53 явно сформулировано: целая функция, пропускающая два значения, постоянна. Значение константы этим не задаётся.", ""),
        ("comparison", "В окрестности a имеем f(z)=(z-a)^3 h(z), где h голоморфна и h(a)≠0. Какие характеристики верны?", [
            "a — нуль f порядка 3.", "1/f имеет в a устранимую особенность.", "1/f имеет в a полюс порядка 3.", "f имеет в a существенно особую точку."], ["A", "C"], [51, 52],
         "Факторизация нуля дана в следствии 8.1; обратная функция равна (z-a)^(-3)/h(z) с ненулевым голоморфным множителем.", ""),
        ("comparison", "Сравните особенности при z=∞ у p(z)=z^4+1 и E(z)=exp(z). Какие утверждения верны?", [
            "У p при бесконечности полюс порядка 4.", "У E при бесконечности устранимая особенность.",
            "У E при бесконечности существенно особая точка.", "У p при бесконечности существенно особая точка."], ["A", "C"], [53],
         "При замене ζ=1/z полином имеет полюс порядка своей степени; экспонента приведена как пример целой трансцендентной функции.", ""),
        ("calculation", "В проколотой окрестности a функция равна 5/(z-a)^3+2/(z-a)+7. Чему равен её вычет в a?", [
            "5", "2", "7", "0"], ["B"], [54],
         "Вычет равен коэффициенту c_{-1} ряда Лорана, то есть 2; коэффициент при степени -3 не является вычетом.", ""),
        ("calculation", "Вычислите вычет f(z)=(z+3)/z^2 в точке 0.", [
            "3", "0", "1", "-1"], ["C"], [54],
         "Разложение f(z)=3/z^2+1/z даёт коэффициент c_{-1}=1.", ""),
        ("insufficient", "Известно только, что у f в a простой полюс. Определите точное значение res_a f; если это невозможно, верните отказ.", [
            "1", "-1", "2", "3"], [], [54],
         "Порядок полюса не определяет ненулевой коэффициент c_{-1}: например, 1/(z-a) и 2/(z-a) дают разные вычеты.",
         "Не задан коэффициент главной части или формула f."),
        ("insufficient", "Известно только, что a — существенно особая точка f. Какое конкретное значение f обязательно пропускает во всех достаточно малых проколотых окрестностях a? Если это не определяется, верните отказ.", [
            "0", "1", "-1", "i"], [], [53],
         "Теорема Пикара допускает не более одного исключительного значения, но не определяет его и не гарантирует, что оно вообще существует.",
         "Нужны сведения о самой функции и её возможном исключительном значении."),
    ]


def prepare(pdf_path, qa_path, output, audit_dir=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Data output must be empty")
    blocks, metadata = extract_pdf(pdf_path, audit_dir)
    digest = metadata["pdf_sha256"]
    questions, golds = parse_qa(Path(qa_path).read_text(encoding="utf-8"), digest)
    own_q, own_g = [], []
    for i, (category, text, choices, answers, pages, rationale, missing) in enumerate(own_examples(), 1):
        qid = f"own_{i:03d}"
        own_q.append(Question(id=qid, question=text, options=[{"id": chr(65 + j), "text": option} for j, option in enumerate(choices)]))
        own_g.append(Gold(id=qid, category=category, answer_ids=answers, abstain=not answers,
                          evidence_sets=[[page_id(digest, p) for p in pages]] if answers else [],
                          rationale=rationale, missing_information=missing))
    ids = {block.block_id for block in blocks}
    for gold in golds + own_g:
        if any(not set(evidence) <= ids for evidence in gold.evidence_sets):
            raise ValueError("Gold points to a missing PDF page")
    for name, rows in [("corpus", blocks), ("dev_questions", questions), ("dev_gold", golds),
                       ("own_questions", own_q), ("own_gold", own_g)]:
        write_jsonl(output / f"{name}.jsonl", rows)
    metadata.update({
        "qa_sha256": sha256_file(qa_path),
        "block_policy": "One source block per nonempty PDF page; ID=pdf_<sha256[:12]>_p<1-based-page:04d>",
        "official_dataset": False, "gold_policy": "All pages explicitly listed by each supplied QA example form one accepted source set; not official block-level gold",
        "math_warning": "Deterministic font/geometry transcription, not a proof of semantic equivalence. Review page_reports, diagrams and complex expressions. No LLM/OCR/enrichment or page-specific formula replacement is used.",
        "files": {p.name: sha256_file(p) for p in sorted(output.glob("*.jsonl"))},
    })
    write_json(output / "data_manifest.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-dir", help="Optional empty directory for all page texts and extraction_audit.json; contains corpus text, keep out of Git")
    args = parser.parse_args()
    result = prepare(args.pdf, args.qa, args.output, args.audit_dir)
    print(f"Extracted {result['pdf_pages']} PDF pages; 15 dev + 8 own questions")
    print(f"Pages flagged for manual review: {len(result['needs_manual_review_pages'])}; see data_manifest.json")


if __name__ == "__main__":
    main()
