"""End-to-end test: load the fine-tuned checkpoint and extract skills.

Needs the weights in `model/best` — download them from the v1.0.0 release:

    https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/tag/v1.0.0
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "model" / "best"
WEIGHTS = MODEL_DIR / "model.safetensors"

pytestmark = pytest.mark.slow

JD_SAMPLE = (
    "Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, C/C++ hoặc Java "
    "(Spring Boot). Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu "
    "PostgreSQL, MySQL, Redis và triển khai Docker, Kubernetes, CI/CD."
)

EXPECTED_SKILLS = [
    "Python",
    "Go",
    "C/C++",
    "Java",
    "Spring Boot",
    "PostgreSQL",
    "MySQL",
    "Redis",
    "Docker",
    "Kubernetes",
    "CI/CD",
]


@pytest.fixture(scope="module")
def extractor():
    if not WEIGHTS.exists():
        pytest.skip(f"weights not found at {WEIGHTS} — download from the v1.0.0 release")
    from src.predict import SkillExtractor

    return SkillExtractor(MODEL_DIR)


def test_extracts_all_skills(extractor):
    skills = extractor.extract_skills(JD_SAMPLE)
    assert skills == EXPECTED_SKILLS


def test_keeps_compound_and_symbol_skills_intact(extractor):
    skills = extractor.extract_skills(JD_SAMPLE)
    for compound in ("C/C++", "Spring Boot", "CI/CD"):
        assert compound in skills


def test_deduplicates_repeated_skills(extractor):
    skills = extractor.extract_skills("Python, Python và Python.")
    assert skills == ["Python"]
