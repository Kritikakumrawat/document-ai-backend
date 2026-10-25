"""Generate synthetic ID-card images for testing and demos.

Every image is stamped "SYNTHETIC TEST DATA — NOT A REAL ID". The names,
numbers and the cartoon portrait are invented; no real person's data is used.

Examples (run from the project root)::

    python scripts/generate_sample_id.py --all --output-dir sample_data/generated
    python scripts/generate_sample_id.py --style badge --scene --output my_badge.jpg
    python scripts/generate_sample_id.py --style license --no-face --output no_face.png
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Printed with a plain hyphen: the bundled font has no em-dash glyph.
WATERMARK = "SYNTHETIC TEST DATA - NOT A REAL ID"

Color = Tuple[int, int, int]


@dataclass
class SyntheticIdSpec:
    style: str = "license"  # "license" (landscape) or "badge" (portrait)
    full_name: str = "ALEX SAMPLE"
    document_number: str = "SYN-4821-7730"
    date_of_birth: date = date(1991, 3, 14)
    expiry_date: date = date(2031, 3, 14)
    issue_date: date = date(2021, 3, 15)
    include_face: bool = True


def _font(size: int) -> ImageFont.ImageFont:
    # Pillow's bundled scalable font: identical output on every OS, no system fonts needed.
    return ImageFont.load_default(size=size)


def _centered_text(draw: ImageDraw.ImageDraw, center_x: int, y: int, text: str, size: int, fill: Color) -> None:
    font = _font(size)
    width = draw.textlength(text, font=font)
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


def draw_synthetic_face(image: Image.Image, box: Tuple[int, int, int, int], include_face: bool = True) -> None:
    """Draw a cartoon portrait (or an empty photo frame) inside ``box``."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    portrait = Image.new("RGB", (width, height), (196, 208, 222))
    draw = ImageDraw.Draw(portrait)

    def span(center_x: float, half_width: float, y0: float, y1: float) -> List[int]:
        """Pixel box centred at ``center_x``; all arguments are fractions of the photo size."""
        left_x, right_x = (center_x - half_width) * width, (center_x + half_width) * width
        return [int(left_x), int(y0 * height), int(right_x), int(y1 * height)]

    if include_face:
        skin: Color = (226, 188, 160)
        shadow: Color = (196, 152, 124)
        hair: Color = (52, 36, 28)
        draw.ellipse(span(0.5, 0.65, 0.80, 1.45), fill=(40, 52, 78))  # shoulders
        draw.rectangle(span(0.5, 0.12, 0.62, 0.86), fill=shadow)  # neck
        draw.ellipse(span(0.5, 0.34, 0.06, 0.52), fill=hair)  # hair behind the head
        for side in (-1, 1):
            draw.ellipse(span(0.5 + side * 0.29, 0.05, 0.34, 0.48), fill=shadow)  # ears
        draw.ellipse(span(0.5, 0.29, 0.14, 0.72), fill=skin)  # face
        draw.chord(span(0.5, 0.30, 0.08, 0.34), 180, 360, fill=hair)  # fringe
        for side in (-1, 1):
            eye_x = 0.5 + side * 0.12
            draw.rectangle(span(eye_x, 0.08, 0.315, 0.335), fill=(45, 32, 26))  # eyebrow
            draw.ellipse(span(eye_x, 0.085, 0.34, 0.415), fill=shadow)  # eye socket
            draw.ellipse(span(eye_x, 0.06, 0.36, 0.40), fill=(235, 235, 235))  # eye white
            draw.ellipse(span(eye_x, 0.028, 0.358, 0.402), fill=(70, 45, 30))  # iris
            draw.ellipse(span(eye_x, 0.012, 0.37, 0.39), fill=(10, 10, 10))  # pupil
        draw.line([int(0.5 * width), int(0.40 * height), int(0.48 * width), int(0.53 * height)], fill=shadow, width=3)
        draw.ellipse(span(0.5, 0.05, 0.52, 0.56), fill=shadow)  # nostrils
        draw.chord(span(0.5, 0.09, 0.57, 0.65), 0, 180, fill=(170, 80, 80))  # mouth
        portrait = portrait.filter(ImageFilter.GaussianBlur(radius=1.5))
    else:
        _centered_text(draw, width // 2, height // 2 - 14, "NO PHOTO", 26, (120, 130, 145))

    image.paste(portrait, (left, top))
    ImageDraw.Draw(image).rectangle(box, outline=(90, 100, 115), width=3)


def _render_license(spec: SyntheticIdSpec) -> Image.Image:
    width, height = 1012, 638
    card = Image.new("RGB", (width, height), (246, 247, 250))
    draw = ImageDraw.Draw(card)
    for y in range(110, 550, 22):  # faint security lines
        draw.line([(320, y), (width - 30, y + 8)], fill=(236, 239, 245), width=2)
    draw.rectangle([0, 0, width, 92], fill=(26, 56, 112))
    draw.text((30, 14), "SYNTHETIC DRIVER LICENSE", font=_font(44), fill=(255, 255, 255))
    draw.text((32, 64), "TEST ISSUING AUTHORITY", font=_font(20), fill=(200, 215, 240))
    draw_synthetic_face(card, (40, 120, 300, 450), spec.include_face)

    rows = [
        f"NAME: {spec.full_name}",
        f"ID NO: {spec.document_number}",
        f"DOB: {spec.date_of_birth:%d/%m/%Y}",
        f"ISS: {spec.issue_date:%d/%m/%Y}",
        f"EXP: {spec.expiry_date:%d/%m/%Y}",
        "CLASS: C",
    ]
    for i, text in enumerate(rows):
        draw.text((330, 128 + i * 64), text, font=_font(36), fill=(20, 24, 32))

    draw.rectangle([0, height - 78, width, height], fill=(196, 30, 36))
    _centered_text(draw, width // 2, height - 62, WATERMARK, 36, (255, 255, 255))
    return card


def _render_badge(spec: SyntheticIdSpec) -> Image.Image:
    width, height = 638, 1012
    card = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, width, 140], fill=(20, 110, 90))
    _centered_text(draw, width // 2, 22, "ACME SYNTHETIC CORP", 40, (255, 255, 255))
    _centered_text(draw, width // 2, 80, "EMPLOYEE BADGE", 30, (210, 240, 230))
    draw_synthetic_face(card, (179, 170, 459, 510), spec.include_face)

    _centered_text(draw, width // 2, 535, spec.full_name, 50, (20, 24, 32))
    rows = [
        f"EMPLOYEE ID: {spec.document_number}",
        "DEPT: RESEARCH",
        f"DATE OF BIRTH: {spec.date_of_birth:%d %b %Y}".upper(),
        f"VALID UNTIL: {spec.expiry_date:%d %b %Y}".upper(),
    ]
    for i, text in enumerate(rows):
        draw.text((48, 622 + i * 54), text, font=_font(32), fill=(20, 24, 32))

    draw.rectangle([0, height - 120, width, height], fill=(196, 30, 36))
    _centered_text(draw, width // 2, height - 108, "SYNTHETIC TEST DATA", 36, (255, 255, 255))
    _centered_text(draw, width // 2, height - 60, "NOT A REAL ID", 36, (255, 255, 255))
    return card


def render_card(spec: SyntheticIdSpec) -> np.ndarray:
    """Return the flat card as a BGR image (the format OpenCV uses)."""
    card = _render_badge(spec) if spec.style == "badge" else _render_license(spec)
    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


def place_in_scene(card: np.ndarray, seed: int = 7, canvas: Tuple[int, int] = (1600, 1200)) -> np.ndarray:
    """Simulate a phone photo: card tilted in perspective on a textured table."""
    rng = np.random.default_rng(seed)
    canvas_w, canvas_h = canvas
    x = np.linspace(0, 1, canvas_w)[None, :]
    y = np.linspace(0, 1, canvas_h)[:, None]
    wood = 70 + 18 * np.sin(40 * y + 6 * np.sin(5 * x)) + 10 * x
    background = np.stack([wood * 0.75, wood * 0.9, wood * 1.15], axis=-1)
    background += rng.normal(0, 6, background.shape)

    card_h, card_w = card.shape[:2]
    scale = 0.62 * min(canvas_w, canvas_h) / max(card_w, card_h) * (1.3 if card_w > card_h else 1.0)
    w, h = card_w * scale, card_h * scale
    cx, cy = canvas_w / 2 + rng.uniform(-60, 60), canvas_h / 2 + rng.uniform(-40, 40)
    angle = np.deg2rad(rng.uniform(-9, 9))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ rotation.T
    corners += np.array([cx, cy]) + rng.uniform(-0.04, 0.04, (4, 2)) * np.array([w, h])

    source = np.array([[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(card, matrix, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((card_h, card_w), 255, np.uint8), matrix, (canvas_w, canvas_h))
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    scene = warped.astype(np.float32) * alpha + background * (1 - alpha)
    lighting = 0.88 + 0.2 * x[..., None] * np.ones_like(y)[..., None]  # gentle light gradient
    scene = np.clip(scene * lighting + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(scene, (3, 3), 0)


def generate_synthetic_id(
    spec: Optional[SyntheticIdSpec] = None, scene: bool = False, seed: int = 7
) -> np.ndarray:
    card = render_card(spec or SyntheticIdSpec())
    return place_in_scene(card, seed=seed) if scene else card


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if extension in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image as {extension}")
    return buffer.tobytes()


def badge_spec(include_face: bool = True) -> SyntheticIdSpec:
    return SyntheticIdSpec(
        style="badge",
        full_name="JORDAN TESTER",
        document_number="EMP-004521",
        date_of_birth=date(1988, 8, 2),
        expiry_date=date(2027, 12, 31),
        include_face=include_face,
    )


STANDARD_SET = {
    "synthetic_license_flat.png": (SyntheticIdSpec(), False),
    "synthetic_license_photo.jpg": (SyntheticIdSpec(), True),
    "synthetic_badge_flat.png": (badge_spec(), False),
    "synthetic_badge_photo.jpg": (badge_spec(), True),
    "synthetic_license_no_face.png": (SyntheticIdSpec(include_face=False), False),
}


def _write(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(image, path.suffix.lower()))
    print(f"wrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic (fake) ID-card images.")
    parser.add_argument("--all", action="store_true", help="write the standard set of test images")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data/generated"))
    parser.add_argument("--output", type=Path, help="output file for a single image (.png or .jpg)")
    parser.add_argument("--style", choices=("license", "badge"), default="license")
    parser.add_argument("--scene", action="store_true", help="tilt the card on a background like a phone photo")
    parser.add_argument("--no-face", action="store_true", help="leave the photo box empty")
    parser.add_argument("--name", help="full name printed on the card")
    parser.add_argument("--number", help="document number printed on the card")
    parser.add_argument("--seed", type=int, default=7, help="random seed for the scene layout")
    args = parser.parse_args(argv)

    if args.all:
        for filename, (spec, scene) in STANDARD_SET.items():
            _write(generate_synthetic_id(spec, scene=scene, seed=args.seed), args.output_dir / filename)
        return 0

    spec = badge_spec() if args.style == "badge" else SyntheticIdSpec()
    spec.include_face = not args.no_face
    if args.name:
        spec.full_name = args.name.upper()
    if args.number:
        spec.document_number = args.number.upper()
    output = args.output or args.output_dir / f"synthetic_{args.style}{'_photo.jpg' if args.scene else '.png'}"
    _write(generate_synthetic_id(spec, scene=args.scene, seed=args.seed), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
