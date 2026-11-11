import pytest

from rag.retrieval import Chunk
from rag.schemas import Gold
from retrieval_diagnostics import PROFILES, diagnose


@pytest.mark.parametrize("profile", list(PROFILES))
def test_diagnostic_modes_neighbors_and_exact_source_spans(profile, question):
    class FakeRetriever:
        chunks = [Chunk("b1", 1, "section", 0, 3, "abc"),
                  Chunk("b2", 2, "section", 2, 5, "x+y")]

        def search(self, q, mode):
            assert q.id == question.id
            assert mode == PROFILES[profile][0]
            return [0]

        def context_order(self, ranked, neighbors):
            assert ranked == [0]
            return [1, 0] if neighbors else [0]

    gold = Gold(id=question.id, category="conditions", answer_ids=["choice-z"],
                abstain=False, evidence_sets=[["b2"]], rationale="Fixture")
    mode, neighbors = PROFILES[profile]
    report = diagnose(FakeRetriever(), [question], {gold.id: gold},
                      mode=mode, neighbors=neighbors, context_chunks=1)
    assert report["mean_page_evidence_recall"] == int(neighbors)
    fragment = report["questions"][0]["candidate_context_fragments"][0]
    assert fragment["text"] == ("x+y" if neighbors else "abc")
    assert fragment["start"] == (2 if neighbors else 0)
    assert report["full_page_evidence_questions"] == int(neighbors)
