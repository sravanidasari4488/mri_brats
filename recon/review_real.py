"""Radiologist review PNGs and a combined PDF for real MRI studies."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recon.ingest_real import DEFAULT_OUTPUT
from recon.qc_real import qc_study
from recon.slices import ORIENT_LETTERS, brain_window, orient_ras_slice, render_overlay_png, window_to_uint8

REVIEW_DIR = ROOT / "reports" / "review"
REVIEW_PDF = ROOT / "reports" / "review_sheet.pdf"
SUMMARY_CSV = ROOT / "reports" / "real_full_summary.csv"
COMPONENT_STUDY = "study_878046c84410"
PAGE_BG = (245, 245, 245)
LEGEND = (
    ((245, 197, 66), "edema"),
    ((192, 57, 43), "necrosis"),
    ((59, 130, 246), "enhancing"),
)


def _load_nifti(path: Path):
    if not path.is_file():
        return None, None
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    return data, np.asarray(image.affine, dtype=np.float64)


def load_display_volume(study_id: str, role: str, *, out_root: Path = DEFAULT_OUTPUT):
    """Prefer 02_registered (pre-z-score); fall back to 04_ready."""
    study = Path(out_root) / study_id
    for folder in ("02_registered", "04_ready"):
        data, affine = _load_nifti(study / folder / f"{role}.nii.gz")
        if data is not None:
            return data, affine
    return None, None


def brain_si_slices(mask, fractions=(0.30, 0.50, 0.70)) -> list[int]:
    """Axial indices at fractions of the brain mask's superior-inferior extent."""
    mask = np.asarray(mask) > 0
    if mask.ndim != 3 or not np.any(mask):
        return [0] * len(fractions)
    zs = np.where(mask.any(axis=(0, 1)))[0]
    lo, hi = int(zs.min()), int(zs.max())
    if hi <= lo:
        return [lo] * len(fractions)
    return [int(round(lo + float(frac) * (hi - lo))) for frac in fractions]


def brightest_flair_slice(flair, mask) -> int:
    """Axial index of the brightest FLAIR voxel inside the brain mask."""
    flair = np.asarray(flair, dtype=np.float32)
    inside = (np.asarray(mask) > 0) & np.isfinite(flair)
    if not np.any(inside):
        return int(flair.shape[2]) // 2
    masked = np.where(inside, flair, -np.inf)
    return int(np.unravel_index(int(np.argmax(masked)), masked.shape)[2])


def empty_pred_slices(flair, mask) -> list[int]:
    """30/50/70% of brain SI, plus the slice with the highest in-brain FLAIR."""
    picks = brain_si_slices(mask)
    bright = brightest_flair_slice(flair, mask)
    if bright not in picks:
        picks.append(bright)
    return sorted(picks)


def wt_peak_slices(label, n: int = 3) -> list[int]:
    wt = np.isin(np.rint(np.asarray(label)).astype(np.int16), (1, 2, 3, 4))
    if wt.ndim != 3:
        return []
    areas = wt.sum(axis=(0, 1))
    if not np.any(areas):
        return []
    order = np.argsort(areas)[::-1]
    picks = []
    for index in order:
        if areas[index] <= 0:
            break
        picks.append(int(index))
        if len(picks) >= n:
            break
    return sorted(picks)


def _font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "C:\\Windows\\Fonts\\arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _gray_masked(volume, mask, index: int, lo: float, hi: float) -> np.ndarray:
    slab = orient_ras_slice(volume, 2, index)
    gray = window_to_uint8(slab, lo, hi)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if mask is not None:
        keep = orient_ras_slice(np.asarray(mask) > 0, 2, index)
        rgb[~keep] = 0
    return rgb


def _overlay_contours(label, index: int, base: np.ndarray) -> np.ndarray:
    if label is None:
        return base
    overlay = Image.open(io.BytesIO(render_overlay_png(label, 2, index))).convert("RGBA")
    canvas = Image.fromarray(base, mode="RGB").convert("RGBA")
    if overlay.size != canvas.size:
        overlay = overlay.resize(canvas.size, Image.NEAREST)
    return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB"))


def _hot_rgb(prob) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros(p.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(p * 3.0, 0.0, 1.0)
    rgb[..., 1] = np.clip(p * 3.0 - 1.0, 0.0, 1.0)
    rgb[..., 2] = np.clip(p * 3.0 - 2.0, 0.0, 1.0)
    return rgb


def _heatmap_on_flair(flair, mask, prob, index: int, lo: float, hi: float) -> np.ndarray:
    base = _gray_masked(flair, mask, index, lo, hi).astype(np.float32)
    if prob is None:
        return base.astype(np.uint8)
    p = orient_ras_slice(np.asarray(prob, dtype=np.float32), 2, index)
    keep = orient_ras_slice(np.asarray(mask) > 0, 2, index) if mask is not None else np.ones(p.shape, dtype=bool)
    color = _hot_rgb(p) * 255.0
    alpha = np.clip(p, 0.0, 1.0) * 0.75
    alpha = np.where(keep, alpha, 0.0)[..., None]
    blended = base * (1.0 - alpha) + color * alpha
    out = np.clip(blended, 0, 255).astype(np.uint8)
    if mask is not None:
        out[~keep] = 0
    return out


def _annotate_slice(rgb: np.ndarray, *, z: int, letters: bool = True) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(14)
    small = _font(12)
    draw.text((4, 2), f"z={z}", fill=(230, 230, 230), font=small)
    if letters:
        w, h = image.size
        axial = ORIENT_LETTERS[2]
        draw.text((4, h // 2 - 8), axial["left"], fill=(255, 220, 80), font=font)
        tw = draw.textlength(axial["right"], font=font)
        draw.text((w - tw - 4, h // 2 - 8), axial["right"], fill=(255, 220, 80), font=font)
        draw.text((w // 2 - 4, 2), axial["top"], fill=(255, 220, 80), font=font)
        draw.text((w // 2 - 4, h - 16), axial["bottom"], fill=(255, 220, 80), font=font)
    return image


def _ijk_to_axial_xy(ijk, shape, index: int) -> tuple[int, int] | None:
    marker = np.zeros(shape[:3], dtype=np.uint8)
    i = int(np.clip(round(float(ijk[0])), 0, shape[0] - 1))
    j = int(np.clip(round(float(ijk[1])), 0, shape[1] - 1))
    marker[i, j, int(index)] = 1
    slab = orient_ras_slice(marker, 2, index)
    ys, xs = np.nonzero(slab)
    if ys.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def _mark_components(rgb: np.ndarray, components: list[dict], index: int, shape, *, slices: list[int] | None = None) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(13)
    for part in components:
        mask = np.asarray(part.get("mask")) if part.get("mask") is not None else None
        if mask is None:
            continue
        sl = mask[:, :, int(index)]
        if np.any(sl):
            coords = np.argwhere(sl)
            ijk = (float(coords[:, 0].mean()), float(coords[:, 1].mean()), float(index))
        else:
            # Still label the nearest shown slice so every table row has a mark.
            if slices:
                centroid_z = float(part["centroid_ijk"][2])
                nearest = min(slices, key=lambda z: abs(z - centroid_z))
                if int(index) != int(nearest):
                    continue
            ijk = (float(part["centroid_ijk"][0]), float(part["centroid_ijk"][1]), float(index))
        xy = _ijk_to_axial_xy(ijk, shape, index)
        if xy is None:
            continue
        x, y = xy
        rank = str(part["rank"])
        tw = draw.textlength(rank, font=font)
        box = [x - tw / 2 - 3, y - 8, x + tw / 2 + 3, y + 8]
        draw.ellipse(box, fill=(20, 20, 20), outline=(255, 255, 255))
        draw.text((x - tw / 2, y - 7), rank, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _study_components(study_id: str, label, affine, wt_prob) -> list[dict]:
    from recon.qc_real import describe_wt_components

    probs = None
    if wt_prob is not None:
        from recon.infer import CHANNEL_ORDER

        stacked = np.zeros((len(CHANNEL_ORDER),) + tuple(wt_prob.shape), dtype=np.float32)
        stacked[CHANNEL_ORDER.index("WT")] = wt_prob
        probs = stacked
    rows = describe_wt_components(label, affine, probs=probs, parc=None)
    from recon.evaluate import regions_from_label
    from recon.qc_real import labeled_wt_components

    parts = labeled_wt_components(regions_from_label(label)["WT"], affine)
    by_id = {part["id"]: part for part in parts}
    for row in rows:
        src = by_id.get(row["id"])
        if src is not None:
            row["mask"] = src["mask"]
    return rows


def _draw_wrapped(draw, text: str, xy, font, fill, max_width: float, line_h: int = 18) -> int:
    words = str(text).split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def render_summary_page(rows: list[dict], size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, PAGE_BG)
    draw = ImageDraw.Draw(canvas)
    title = _font(22)
    head = _font(13)
    body = _font(12)
    draw.text((24, 16), "Brain MRI review sheet — 12 hospital exams", fill=(20, 20, 20), font=title)
    notes = [
        "What the study ID is. Each study_xxxxxxxxxxxx is an anonymized code for one MRI exam. "
        "It is a hash of the DICOM Study Instance UID. It is not the patient's name, hospital ID, date of birth, or the folder name on the scanner export. "
        "Please use only this code if you refer to a case.",
        "What this packet is. These 12 exams had FLAIR, pre-contrast T1, post-contrast T1 (T1c), and T2. "
        "Later pages are one exam each. Axial images use radiological convention: the patient's left is on your right (R on the left edge of each slice).",
        "What the overlays are. Colored outlines are a research model's prediction, not a radiologist contour: yellow = edema, red = necrosis, blue = enhancing tumor. "
        "FLAIR pWT is the model's whole-tumor probability (warmer = more sure). QC PASS = technically usable; WARN = usable but thick slices or similar; FAIL = do not use a prediction (missing sequence or failed alignment) — no model contours are shown.",
        "What to do. For every exam, mark tumor / no tumor / other in the box on that page, and fill the Radiologist diagnosis column below. Volumes (cm3) are model estimates for reference only.",
    ]
    y = 50
    max_w = size[0] - 48
    for note in notes:
        y = _draw_wrapped(draw, note, (24, y), body, (40, 40, 40), max_w, 17)
        y += 8
    y += 6
    headers = ("study_id", "QC", "WT cm3", "TC cm3", "ET cm3", "ABC/2", "n", "Radiologist diagnosis")
    keys = ("study_id", "qc_status", "wt_cm3", "tc_cm3", "et_cm3", "abc2_largest_cm3", "wt_components", "")
    col_w = [210, 70, 80, 80, 80, 80, 40, max(220, size[0] - 24 * 2 - 640)]
    x0, y0 = 24, y
    row_h = 36
    x = x0
    for header, width in zip(headers, col_w):
        draw.rectangle([x, y0, x + width, y0 + row_h], outline=(80, 80, 80), fill=(230, 230, 230))
        draw.text((x + 6, y0 + 10), header, fill=(20, 20, 20), font=head)
        x += width
    for i, row in enumerate(rows):
        y = y0 + (i + 1) * row_h
        x = x0
        fill = (255, 255, 255) if i % 2 == 0 else (236, 236, 236)
        qc = (row.get("qc_status") or "").upper()
        if qc == "FAIL":
            fill = (255, 228, 228)
        elif qc == "WARN":
            fill = (255, 244, 220)
        for key, width in zip(keys, col_w):
            draw.rectangle([x, y, x + width, y + row_h], outline=(160, 160, 160), fill=fill)
            text = "" if key == "" else str(row.get(key) or "")
            draw.text((x + 6, y + 10), text[:28], fill=(20, 20, 20), font=body)
            x += width
    return canvas


def render_study_review(
    study_id: str,
    *,
    out_root: Path = DEFAULT_OUTPUT,
    dest: Path | None = None,
    qc=None,
) -> Path:
    out_root = Path(out_root)
    if qc is None:
        qc = qc_study(study_id, out_root=out_root)
    failed = bool(qc.failed)
    ready = out_root / study_id / "04_ready"
    flair, _ = load_display_volume(study_id, "flair", out_root=out_root)
    t1c, _ = load_display_volume(study_id, "t1c", out_root=out_root)
    mask, _ = _load_nifti(ready / "brain_mask.nii.gz")
    label = None
    affine = None
    if not failed:
        label, affine = _load_nifti(out_root / study_id / "pred_seg.nii.gz")
    wt_prob, _ = (None, None) if failed else _load_nifti(out_root / study_id / "wt_prob.nii.gz")
    if flair is None and t1c is None:
        raise FileNotFoundError(f"No FLAIR or T1c for {study_id}")
    reference = flair if flair is not None else t1c
    if mask is None:
        mask = np.abs(reference) > 0
    wt_slices = wt_peak_slices(label) if label is not None else []
    if wt_slices:
        slices = wt_slices
    else:
        slices = empty_pred_slices(flair if flair is not None else reference, mask)

    overlay_label = None if failed else label
    components = []
    if study_id == COMPONENT_STUDY and overlay_label is not None and affine is not None:
        components = _study_components(study_id, overlay_label, affine, wt_prob)

    flair_lo = flair_hi = 0.0
    t1c_lo = t1c_hi = 0.0
    if flair is not None:
        flair_lo, flair_hi = brain_window(flair, mask)
    if t1c is not None:
        t1c_lo, t1c_hi = brain_window(t1c, mask)

    rows = []
    captions = []
    if flair is not None:
        cells = []
        for index in slices:
            rgb = _gray_masked(flair, mask, index, flair_lo, flair_hi)
            rgb = _overlay_contours(overlay_label, index, rgb)
            if components:
                rgb = _mark_components(rgb, components, index, flair.shape, slices=slices)
            cells.append(_annotate_slice(rgb, z=index))
        rows.append(cells)
        captions.append("FLAIR")
    if t1c is not None:
        cells = []
        for index in slices:
            rgb = _gray_masked(t1c, mask, index, t1c_lo, t1c_hi)
            rgb = _overlay_contours(overlay_label, index, rgb)
            cells.append(_annotate_slice(rgb, z=index))
        rows.append(cells)
        captions.append("T1c")
    if flair is not None and not failed:
        cells = []
        for index in slices:
            rgb = _heatmap_on_flair(flair, mask, wt_prob, index, flair_lo, flair_hi)
            cells.append(_annotate_slice(rgb, z=index))
        rows.append(cells)
        captions.append("FLAIR pWT")

    cell_w, cell_h = rows[0][0].size
    pad = 8
    header = 58
    legend_h = 28
    table_h = 22 * (len(components) + 2) if components else 0
    footer = 58
    width = pad + len(slices) * (cell_w + pad)
    height = header + legend_h + footer + table_h + len(rows) * (cell_h + 20 + pad)
    canvas = Image.new("RGB", (width, max(height, 200)), PAGE_BG)
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(13)
    qc_text = f"{study_id}    QC {qc.status}"
    if qc.reasons:
        qc_text += "  —  " + "; ".join(qc.reasons)[:90]
    draw.text((pad, 6), qc_text, fill=(20, 20, 20), font=font)
    draw.text(
        (pad, 28),
        "Anonymized exam ID (hash of the DICOM study). Not a patient name or folder name. See page 1.",
        fill=(80, 80, 80),
        font=small,
    )
    lx = pad
    for color, name in LEGEND:
        draw.rectangle([lx, header - 2, lx + 14, header + 12], fill=color, outline=(40, 40, 40))
        draw.text((lx + 18, header - 2), name, fill=(40, 40, 40), font=small)
        lx += 90
    y = header + legend_h
    for caption, images in zip(captions, rows):
        draw.text((pad, y), caption, fill=(40, 40, 40), font=small)
        y += 16
        x = pad
        for image in images:
            canvas.paste(image, (x, y))
            x += cell_w + pad
        y += cell_h + pad
    if components:
        draw.text((pad, y), "WT components (rank, volume, TC, mean pWT)", fill=(20, 20, 20), font=small)
        y += 18
        for part in components:
            mean_p = part.get("mean_wt_prob")
            mean_s = "-" if mean_p is None else f"{float(mean_p):.3f}"
            tc = "yes" if part.get("contains_tc") else "no"
            line = f"#{part['rank']}  {part['volume_cm3']:.2f} cm3  TC {tc}  pWT {mean_s}"
            draw.text((pad, y), line, fill=(20, 20, 20), font=small)
            y += 16
        y += 4
    box = [pad, height - footer + 6, width - pad, height - 8]
    draw.rectangle(box, outline=(80, 80, 80), width=1, fill=(255, 255, 255))
    draw.text((pad + 8, height - footer + 16), "Radiologist: tumor / no tumor / other", fill=(40, 40, 40), font=small)
    dest = dest or (REVIEW_DIR / f"{study_id}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


def _load_summary_rows(path: Path = SUMMARY_CSV) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_review_pdf(pngs: list[Path], dest: Path = REVIEW_PDF, *, summary_rows: list[dict] | None = None) -> Path:
    study_pages = [Image.open(path).convert("RGB") for path in pngs]
    if not study_pages:
        raise ValueError("No review PNGs to combine")
    width = max(page.size[0] for page in study_pages)
    height = max(page.size[1] for page in study_pages)
    width = max(width, 1100)
    height = max(height, 720)
    pages = []
    rows = summary_rows if summary_rows is not None else _load_summary_rows()
    if rows:
        pages.append(render_summary_page(rows, (width, height)))
    for page in study_pages:
        canvas = Image.new("RGB", (width, height), PAGE_BG)
        canvas.paste(page, (0, 0))
        pages.append(canvas)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(dest, save_all=True, append_images=pages[1:], resolution=120.0)
    return dest


def render_tier_reviews(study_ids, *, out_root: Path = DEFAULT_OUTPUT) -> tuple[list[Path], Path]:
    pngs = []
    rows = _load_summary_rows()
    by_id = {row.get("study_id"): row for row in rows}
    ordered = list(study_ids)
    for study_id in ordered:
        try:
            pngs.append(render_study_review(study_id, out_root=out_root))
            print(f"  review {study_id}", flush=True)
        except Exception as exc:
            print(f"  review skip {study_id}: {exc}", flush=True)
    summary = [by_id[study_id] for study_id in ordered if study_id in by_id]
    if not summary:
        summary = rows
    pdf = write_review_pdf(pngs, summary_rows=summary) if pngs else REVIEW_PDF
    return pngs, pdf


def export_wt_prob(study_id: str, *, out_root: Path = DEFAULT_OUTPUT, network=None, checkpoint=None) -> Path | None:
    """Write wt_prob.nii.gz from a fast TTA pass. Skips QC-FAIL studies."""
    from recon.infer import CHANNEL_ORDER, apply_model_preprocessing, load_model, predict_probs, resample_probabilities, resolve_checkpoint, roi_for
    from recon.io import load_case

    qc = qc_study(study_id, out_root=out_root)
    if qc.failed:
        return None
    dest = Path(out_root) / study_id / "wt_prob.nii.gz"
    if dest.is_file():
        return dest
    ready = Path(out_root) / study_id / "04_ready"
    loaded = load_case(ready, normalize=False)
    affine = np.asarray(loaded["affine"], dtype=np.float64)
    if network is None:
        checkpoint = checkpoint or resolve_checkpoint("abhinav_v3")
        network = load_model(checkpoint, None, "abhinav_v3")
    network_image, network_affine = apply_model_preprocessing(loaded["image"], affine, "abhinav_v3")
    probs = predict_probs(network, network_image, roi_for("abhinav_v3"), "fast", model_name="abhinav_v3", overlap=0.5)
    if probs.shape[1:] != tuple(loaded["image"].shape[1:]):
        probs = resample_probabilities(probs, network_affine, loaded["image"].shape[1:], affine)
    wt = np.asarray(probs[CHANNEL_ORDER.index("WT")], dtype=np.float32)
    nib.save(nib.Nifti1Image(wt, affine), str(dest))
    return dest


if __name__ == "__main__":
    import argparse

    from recon.ingest_real import load_inventory_rows
    from scripts.inventory_real import DEFAULT_OUT

    parser = argparse.ArgumentParser(description="Render radiologist review sheets")
    parser.add_argument("--tier", default="FULL")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--studies", nargs="*", default=None)
    parser.add_argument("--predict-missing", action="store_true", help="Write wt_prob.nii.gz when missing")
    args = parser.parse_args()
    rows = load_inventory_rows(args.inventory, args.tier)
    ids = [row["study_id"] for row in rows if row.get("study_id")]
    if args.studies:
        keep = set(args.studies)
        ids = [study_id for study_id in ids if study_id in keep]
    if args.predict_missing:
        from recon.infer import load_model, resolve_checkpoint

        checkpoint = resolve_checkpoint("abhinav_v3")
        network = None
        for study_id in ids:
            dest = Path(args.out) / study_id / "wt_prob.nii.gz"
            qc = qc_study(study_id, out_root=args.out)
            if qc.failed or dest.is_file():
                continue
            if network is None:
                print("  loading model for wt_prob", flush=True)
                network = load_model(checkpoint, None, "abhinav_v3")
            print(f"  wt_prob {study_id}", flush=True)
            export_wt_prob(study_id, out_root=args.out, network=network, checkpoint=checkpoint)
    pngs, pdf = render_tier_reviews(ids, out_root=args.out)
    print(f"Wrote {len(pngs)} PNGs and {pdf}", flush=True)
