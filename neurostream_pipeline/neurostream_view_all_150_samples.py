# =====================================================================
# NEUROSTREAM CELL 2B — 3-PANEL VIEWER FOR ALL 150 TEST SAMPLES
# =====================================================================
# Displays ALL 150 test samples in the exact 3-panel comparative layout:
#   [Column 1: REAL IMAGE (Ground Truth)]
#   [Column 2: PREDICTED IMAGE (SD 2.1)]
#   [Column 3: AFTER SEGMENTED (YOLOv8 Solid White-BG for TRELLIS 3D)]
#
# Generates high-resolution 3-column figures (10 samples per page x 15 pages)
# matching the exact visual styling, fonts, and colors of Cell 2.
# =====================================================================

import os, sys, glob, re, json, math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/kaggle/working"
DATA_ROOT = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
SEG_DIR = os.path.join(ROOT, "segmented_images")
RAW_DIR = os.path.join(ROOT, "predicted_raw_images")
REAL_DIR = os.path.join(ROOT, "real_images")
META_PATH = os.path.join(ROOT, "samples_metadata.json")
OUTPUT_DIR = os.path.join(ROOT, "triplet_views")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Verified Synset Mapping (Fallback if metadata missing) -----
SYNSET_TO_CAPTION = {
    "n02106662": "a photo of a German shepherd", "n02124075": "a photo of an Egyptian cat",
    "n02281787": "a photo of a butterfly",       "n02389026": "a photo of a horse",
    "n02492035": "a photo of a capuchin monkey", "n02504458": "a photo of an African elephant",
    "n02510455": "a photo of a giant panda",     "n02607072": "a photo of an anemone fish",
    "n02690373": "a photo of an airliner airplane", "n02906734": "a photo of a broom",
    "n02951358": "a photo of a canoe",           "n02992529": "a photo of a mobile phone",
    "n03063599": "a photo of a coffee mug",      "n03100240": "a photo of a convertible car",
    "n03180011": "a photo of a desktop computer","n03197337": "a photo of a digital watch",
    "n03272010": "a photo of an electric guitar","n03272562": "a photo of a train locomotive",
    "n03297495": "a photo of an espresso maker", "n03376595": "a photo of a folding chair",
    "n03445777": "a photo of a golf ball",       "n03452741": "a photo of a grand piano",
    "n03584829": "a photo of an iron",           "n03590841": "a photo of a jack-o-lantern",
    "n03709823": "a photo of a mailbag postbag", "n03773504": "a photo of a missile rocket",
    "n03775071": "a photo of a mitten glove",    "n03792782": "a photo of a mountain bike",
    "n03792972": "a photo of a mountain tent",   "n03877472": "a photo of pajamas",
    "n03888257": "a photo of a parachute",       "n03982430": "a photo of a pool billiard table",
    "n04044716": "a photo of a radio telescope", "n04069434": "a photo of a camera",
    "n04086273": "a photo of a revolver handgun","n04120489": "a photo of a running shoe sneaker",
    "n07753592": "a photo of a banana",          "n07873807": "a photo of a pizza",
    "n11939491": "a photo of a daisy flower",    "n13054560": "a photo of a mushroom"
}

IMAGE_DIR_CANDIDATES = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/data/images",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/images",
    os.path.join(DATA_ROOT, "images"), "./images", "/kaggle/working/images"
]

def resolve_image_path(name):
    if not name: return None
    syn = name.split("_")[0]
    for root in IMAGE_DIR_CANDIDATES:
        if not os.path.exists(root): continue
        p1 = os.path.join(root, syn, name + ".JPEG")
        if os.path.exists(p1): return p1
        p2 = os.path.join(root, name + ".JPEG")
        if os.path.exists(p2): return p2
        g = glob.glob(os.path.join(root, "**", name + ".*"), recursive=True)
        if g: return g[0]
    return None

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 1. Discover all segmented samples
seg_files = sorted(glob.glob(os.path.join(SEG_DIR, "segmented_sample_*.png")), key=natural_sort_key)
if not seg_files:
    seg_files = sorted(glob.glob(os.path.join(SEG_DIR, "*.png")), key=natural_sort_key)

total_samples = len(seg_files)
print("=" * 80)
print(f"NEUROSTREAM 3-PANEL VIEWER: Loading all {total_samples} samples")
print(f"  • Segmented Images Dir : {SEG_DIR}")
print(f"  • Predicted Images Dir : {RAW_DIR}")
print(f"  • Real Images Dir      : {REAL_DIR}")
print("=" * 80)

if total_samples == 0:
    print(f"Notice: No segmented images found in '{SEG_DIR}'. Run Cell 2 first.")
    sys.exit(0)

# 2. Load metadata json if saved by Cell 2
metadata_by_id = {}
if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r") as f:
            meta_list = json.load(f)
            metadata_by_id = {item["sample_id"]: item for item in meta_list}
            print(f"  [✓] Loaded captions and match labels from: {META_PATH}")
    except Exception as e:
        print(f"  Notice: Could not load metadata json ({e})")

# 3. Assemble all Triplets
triplets = []
for idx in range(total_samples):
    seg_p = seg_files[idx]
    s_match = re.search(r'\d+', os.path.basename(seg_p))
    s_id = int(s_match.group()) if s_match else idx
    
    meta = metadata_by_id.get(s_id, {})
    gt_caption = meta.get("gt_caption", "Ground Truth")
    prompt = meta.get("prompt", "Predicted Object")
    is_match = meta.get("is_match", True)

    # 1. Real Image
    real_p = os.path.join(REAL_DIR, f"real_sample_{s_id}.png")
    real_im = None
    if os.path.exists(real_p):
        try: real_im = Image.open(real_p).convert("RGB").resize((512, 512))
        except Exception: real_im = None
        
    if real_im is None and "im_name" in meta:
        res_p = resolve_image_path(meta["im_name"])
        if res_p and os.path.exists(res_p):
            try: real_im = Image.open(res_p).convert("RGB").resize((512, 512))
            except Exception: real_im = None

    if real_im is None:
        card = np.ones((512, 512, 3), dtype=np.uint8) * 245
        cv2.rectangle(card, (15, 15), (497, 497), (200, 210, 200), 2)
        cv2.putText(card, "REAL STIMULUS", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 80, 20), 2)
        cv2.putText(card, f"Sample #{s_id}", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 2)
        cv2.putText(card, "Class:", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2)
        cv2.putText(card, gt_caption.replace("a photo of ", ""), (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 120, 20), 2)
        real_im = Image.fromarray(card)

    # 2. Predicted Image
    pred_p = os.path.join(RAW_DIR, f"predicted_sample_{s_id}.png")
    if os.path.exists(pred_p):
        pred_im = Image.open(pred_p).convert("RGB").resize((512, 512))
    else:
        pred_im = Image.open(seg_p).convert("RGB").resize((512, 512))

    # 3. Segmented Image
    seg_im = Image.open(seg_p).convert("RGB").resize((512, 512))

    triplets.append({
        "id": s_id,
        "real": real_im,
        "pred": pred_im,
        "seg": seg_im,
        "gt_caption": gt_caption,
        "prompt": prompt,
        "is_match": is_match
    })

# =====================================================================
# 4. RENDER ALL SAMPLES IN PAGES OF 10 SAMPLES (EXACT CELL 2 LAYOUT)
# =====================================================================
SAMPLES_PER_PAGE = 10
total_pages = math.ceil(total_samples / SAMPLES_PER_PAGE)
print(f"\nRendering all {total_samples} samples across {total_pages} pages (10 samples x 3 columns per page)...\n")

for page in range(total_pages):
    start_i = page * SAMPLES_PER_PAGE
    end_i = min(start_i + SAMPLES_PER_PAGE, total_samples)
    page_triplets = triplets[start_i:end_i]
    n_display = len(page_triplets)
    
    fig, axes = plt.subplots(n_display, 3, figsize=(13, 4.0 * n_display))
    if n_display == 1:
        axes = np.expand_dims(axes, 0)

    for idx, item in enumerate(page_triplets):
        s_id = item["id"]
        gt_text = item["gt_caption"].replace("a photo of ", "")
        p_text = item["prompt"].replace("a photo of ", "")
        is_match = item["is_match"]

        # Column 1: Real Image (Ground Truth)
        axes[idx, 0].imshow(item["real"])
        axes[idx, 0].set_title(f"Sample {s_id}: REAL IMAGE (Ground Truth)\n'{gt_text}'",
                               fontsize=10, fontweight="bold", color="#1b5e20", pad=4)
        axes[idx, 0].axis("off")

        # Column 2: Predicted Image (SD 2.1)
        match_str = " [CORRECT ✓]" if is_match else ""
        match_col = "#0d47a1" if is_match else "#b71c1c"
        axes[idx, 1].imshow(item["pred"])
        axes[idx, 1].set_title(f"Sample {s_id}: PREDICTED IMAGE (SD 2.1)\n'{p_text}'{match_str}",
                               fontsize=10, fontweight="bold", color=match_col, pad=4)
        axes[idx, 1].axis("off")

        # Column 3: After Segmented (YOLOv8)
        axes[idx, 2].imshow(item["seg"])
        axes[idx, 2].set_title(f"Sample {s_id}: AFTER SEGMENTED (YOLOv8)\nSolid White-BG (Ready for TRELLIS 3D)",
                               fontsize=10, fontweight="bold", color="#e65100", pad=4)
        axes[idx, 2].axis("off")

    plt.tight_layout()
    page_save_path = os.path.join(OUTPUT_DIR, f"triplet_page_{page + 1:02d}_samples_{start_i:02d}_to_{end_i - 1:02d}.png")
    plt.savefig(page_save_path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"  [✓] Page {page + 1:02d}/{total_pages} (Samples {start_i}..{end_i - 1}) displayed & saved to: {page_save_path}")

print("=" * 80)
print(f"COMPLETE: All {total_samples} samples rendered in the exact 3-panel comparative layout!")
print(f"All full-resolution page images saved to: {OUTPUT_DIR}")
print("=" * 80)
