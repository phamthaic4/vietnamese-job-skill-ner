"""Unit tests for the tokenization / post-processing layer (no model needed)."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import TOKEN_RE, SkillExtractor  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C/C++", ["C/C++"]),
        ("CI/CD", ["CI/CD"]),
        ("C#", ["C#"]),
        ("Java (Spring Boot)", ["Java", "(", "Spring", "Boot", ")"]),
        ("Python, Docker.", ["Python", ",", "Docker."]),
    ],
)
def test_token_re_keeps_symbols_attached(text, expected):
    assert TOKEN_RE.findall(text) == expected


def test_token_re_splits_on_whitespace_and_punctuation():
    assert TOKEN_RE.findall("Spring Boot") == ["Spring", "Boot"]
    assert TOKEN_RE.findall("PostgreSQL, Redis") == ["PostgreSQL", ",", "Redis"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Spring Boot", ["Spring Boot"]),
        ("C/C++", ["C/C++"]),
        ("CI/CD.", ["CI/CD"]),
        (".NET", [".NET"]),
        ("Python,", ["Python"]),
        ("  , ;  ", []),
        ("A", []),
        ("C", ["C"]),
        ("R", ["R"]),
    ],
)
def test_post_process(raw, expected):
    assert SkillExtractor.post_process([raw]) == expected


def test_post_process_deduplicates_case_insensitively_keeping_order():
    assert SkillExtractor.post_process(["Python", "python", "Docker", "Python"]) == [
        "Python",
        "Docker",
    ]


def test_post_process_collapses_whitespace():
    assert SkillExtractor.post_process(["Spring   Boot"]) == ["Spring Boot"]


def test_token_regex_is_compiled_with_unicode_flags():
    assert isinstance(TOKEN_RE, re.Pattern)
    assert TOKEN_RE.flags & re.UNICODE
