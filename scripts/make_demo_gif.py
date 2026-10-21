"""Render the README demo GIF (docs/images/demo_predict.gif).

Draws the real `python src/predict.py` terminal session frame by frame, so the
README demo is reproducible instead of a one-off screen recording.

    python scripts/make_demo_gif.py
"""

from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "images" / "demo_predict.gif"

MPL_FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

# DejaVu Sans Mono is missing several Vietnamese glyphs, so prefer a font that
# covers Latin Extended Additional. First candidate with full coverage wins.
FONT_CANDIDATES = [
    (Path(r"C:\Windows\Fonts\consola.ttf"), Path(r"C:\Windows\Fonts\consolab.ttf")),
    (MPL_FONTS / "DejaVuSansMono.ttf", MPL_FONTS / "DejaVuSansMono-Bold.ttf"),
]

VIETNAMESE = "ẢệảữẤừỆĐỘỳỌỗơưăâđêôẾỆỮ"

W, H = 1000, 620
PAD = 24
LINE_H = 20
CHARS_PER_LINE = 92
COMMAND_LINES = 5

BG = (13, 17, 23)
FG = (230, 237, 243)
DIM = (110, 118, 129)
GREEN = (126, 231, 135)
BLUE = (88, 166, 255)

COMMAND = (
    'python src/predict.py --text "Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, '
    'C/C++ hoặc Java (Spring Boot). Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu '
    'PostgreSQL, MySQL, Redis và triển khai Docker, Kubernetes, CI/CD."'
)

SKILLS = [
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


def wrap(text, width=CHARS_PER_LINE):
    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_steps():
    """Return the animation steps: (kind, text) tuples."""
    steps = []
    for i in range(8, len(COMMAND) + 1, 8):
        steps.append(("cmd", COMMAND[:i]))
    steps.append(("info", "VĂN BẢN ĐẦU VÀO:"))
    for line in wrap(COMMAND[len('python src/predict.py --text "') : -2]):
        steps.append(("dim", line))
    steps.append(("dim", "-" * CHARS_PER_LINE))
    steps.append(("info", "KỸ NĂNG TRÍCH XUẤT ĐƯỢC (11 skills):"))
    for idx, skill in enumerate(SKILLS, start=1):
        steps.append(("skill", f"  {idx}. {skill}"))
    steps.append(("dim", "=" * CHARS_PER_LINE))
    return steps


def _glyph(font, ch):
    im = Image.new("L", (34, 34), 0)
    ImageDraw.Draw(im).text((3, 3), ch, font=font, fill=255)
    return im.tobytes()


def pick_fonts():
    """Return (regular, bold) font paths — the first pair covering Vietnamese."""
    for regular, bold in FONT_CANDIDATES:
        if not regular.exists():
            continue
        font = ImageFont.truetype(str(regular), 15)
        notdef = _glyph(font, "\ue123")
        if all(_glyph(font, ch) != notdef for ch in VIETNAMESE):
            bold_path = bold if bold.exists() else regular
            return regular, bold_path
    fallback = MPL_FONTS / "DejaVuSansMono.ttf"
    return fallback, MPL_FONTS / "DejaVuSansMono-Bold.ttf"


FONT_PATH, BOLD_FONT_PATH = pick_fonts()

COLOR = {"cmd": GREEN, "info": BLUE, "dim": DIM, "skill": FG}


def render(steps):
    font = ImageFont.truetype(str(FONT_PATH), 15)
    bold = ImageFont.truetype(str(BOLD_FONT_PATH), 15)

    frames, durations = [], []
    for i, (kind, _) in enumerate(steps):
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)

        draw.rectangle([0, 0, W, 34], fill=(22, 27, 34))
        for idx, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            draw.ellipse([16 + idx * 20, 12, 28 + idx * 20, 24], fill=color)
        draw.text((W / 2, 17), "python src/predict.py", font=font, fill=DIM, anchor="mm")

        # command block: only the longest prefix so far, fixed height so the
        # output below never shifts
        cmd_prefix = ""
        for j, (k2, t2) in enumerate(steps):
            if j > i or k2 != "cmd":
                break
            cmd_prefix = t2

        y = 34 + PAD
        for line in wrap(cmd_prefix)[:COMMAND_LINES]:
            draw.text((PAD, y), line, font=font, fill=COLOR["cmd"])
            y += LINE_H

        y = 34 + PAD + COMMAND_LINES * LINE_H + 12
        for j, (k2, t2) in enumerate(steps):
            if k2 == "cmd" or j > i:
                continue
            draw.text(
                (PAD, y),
                t2,
                font=bold if k2 == "info" else font,
                fill=COLOR[k2],
            )
            y += LINE_H

        frames.append(image)
        durations.append(40 if kind == "cmd" else 240)

    durations[-1] = 3000
    return frames, durations


def main():
    steps = build_steps()
    frames, durations = render(steps)

    master = frames[-1].quantize(colors=64)
    palette_frames = [f.quantize(palette=master, dither=Image.Dither.NONE) for f in frames]

    palette_frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )

    with Image.open(OUT_PATH) as check:
        written = check.n_frames

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({written} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
