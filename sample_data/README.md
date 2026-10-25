# Sample data

**Only synthetic images belong here. Never add photos or scans of real identity documents.**

`synthetic_license_example.jpg` is a computer-generated card, stamped
"SYNTHETIC TEST DATA - NOT A REAL ID". The name, number, dates and the cartoon
portrait are invented. It simulates a phone photo: the card is tilted in
perspective on a wooden table, so it also exercises card detection.

It was created with (run from the project root):

```bash
python scripts/generate_sample_id.py --scene --seed 4 --output sample_data/synthetic_license_example.jpg
```

## Generate more test images

```bash
python scripts/generate_sample_id.py --all --output-dir sample_data/generated
```

This writes five images to `sample_data/generated/` (ignored by Git):

| File | What it tests |
|------|---------------|
| `synthetic_license_flat.png` | Flat landscape card, labelled fields, DD/MM/YYYY dates |
| `synthetic_license_photo.jpg` | Same card tilted on a background (card detection + perspective correction) |
| `synthetic_badge_flat.png` | Portrait employee badge, unlabelled name, "02 AUG 1988" style dates |
| `synthetic_badge_photo.jpg` | Same badge tilted on a background |
| `synthetic_license_no_face.png` | Empty photo box: face extraction must report `not_detected` |

Other options: `--style badge`, `--scene`, `--no-face`, `--name "TEST PERSON"`,
`--number "XYZ-123456"`, `--seed 11`, `--output path.jpg`. Run
`python scripts/generate_sample_id.py --help` for the full list.
