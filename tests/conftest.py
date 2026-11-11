from __future__ import annotations

import json
import os
import socket
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import pytest

from rag.config import load_settings
from rag.schemas import Block, Question


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not access the network or a real LLM")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def settings():
    return load_settings(Path(__file__).resolve().parents[1] / "config.json")


@pytest.fixture
def question():
    return Question(id="unseen-id", question="Какой вывод следует из условия?", options=[
        {"id": label, "text": text} for label, text in zip(
            ["choice-z", "choice-a", "choice-7", "choice-x"],
            ["Ограниченная целая функция постоянна.", "Любая функция постоянна.", "Производная всегда равна 1.", "Недостаточно непрерывности."]
        )])


@pytest.fixture
def blocks():
    return [
        Block(block_id="original-α", page=1, section="Целые функции", text="По теореме Лиувилля ограниченная целая функция постоянна. " * 10),
        Block(block_id="original-β", page=2, section="Целые функции", text="Голоморфность требует существования комплексной производной. " * 10),
        Block(block_id="original-γ", page=3, section="Другое", text="Интеграл по окружности зависит от ориентации контура. " * 10),
    ]


class CharacterCounter:
    """Deliberately simple deterministic mock, never used in experiments."""
    def count(self, messages):
        return len(json.dumps(messages, ensure_ascii=False)) // 3 + 10

