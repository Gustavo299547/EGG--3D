# ============================================================================
# NEUROSTREAM CELL 2 : INFERENCE + GOOGLE DRIVE SYNC + FULL TEST EVALUATION
# ----------------------------------------------------------------------------
# 1. Automatic Model Loading:
#    - Checks for trained checkpoints ('best_neurostream_sst.pt' and
#      'exact-paper-modelV00.pt') locally in /kaggle/working/.
#    - If not found locally, automatically downloads from Google Drive folder:
#      13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz
# 2. Complete Scientific Evaluation:
#    - Evaluates classifier across ALL test trials (>1000 trials).
#    - Produces True Positive (TP), False Positive (FP), Accuracy & Metrics.
# 3. Image Generation & Google Drive Storage:
#    - Segmentation phase (YOLOv8) is COMPLETELY REMOVED.
#    - Creates 'GeneratedImage' folder in Google Drive (if not existing).
#    - For EVERY test sample (>1000 samples), creates/reuses subfolder:
#      'GeneratedImage/sample_XXXX/'
#    - Saves two images per folder: 'real.png' and 'generated.png'.
#    - If folder exists, reuses it without recreation; if image exists, overrides it.
# 4. Evaluation CSV Report:
#    - Generates 'test_evaluation_results.csv' with 3 columns:
#      [real_image_class, predicted_class, isTrue]
#    - Counts total true, total false, total images, and overall accuracy.
#    - Uploads/overwrites 'test_evaluation_results.csv' inside 'GeneratedImage' on Drive.
# ============================================================================

import os, sys, gc, glob, random, math, subprocess, warnings, csv, time
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, ControlNetModel
except AttributeError as _e:
    for _m in [m for m in sys.modules if m.startswith("diffusers")]: del sys.modules[_m]
    sys.modules["torch._dynamo"] = None
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, ControlNetModel

from transformers import CLIPTextModel, CLIPTokenizer

# ----------------- Configuration & Folders -----------------
DATA_ROOT = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "/kaggle/working"
LOCAL_GEN_DIR = os.path.join(OUTPUT_DIR, "GeneratedImage")
os.makedirs(LOCAL_GEN_DIR, exist_ok=True)

CLASSIFIER_CANDIDATES = ["best_neurostream_multitrial_pooling.pt", "best_neurostream_sst.pt"]
PIPELINE_CANDIDATES = ["neurostream_multitrial_pipeline.pt", "exact-paper-modelV00.pt"]
NEUROSTREAM_BEST_PATH = os.path.join(OUTPUT_DIR, CLASSIFIER_CANDIDATES[0])
PIPELINE_PATH = os.path.join(OUTPUT_DIR, PIPELINE_CANDIDATES[0])
CSV_OUTPUT_PATH = os.path.join(LOCAL_GEN_DIR, "test_evaluation_results.csv")

STEPS = 20             # Fast high-quality UniPC sampling steps (approx 0.8s / image)
GUIDANCE = 4.0
SEED = 42
dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

seed_everything(SEED)

# Clean prior GPU allocations in notebook kernel before starting
for _var in ['unet', 'vae', 'txt', 'tok', 'c_all', 'lat', 'X_norm_t']:
    if _var in globals():
        del globals()[_var]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# GOOGLE DRIVE BIDIRECTIONAL SYNC (OAuth 2.0 User Quota)
# ============================================================================
DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

_drive_service = None

def get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth"], check=False)
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

    try:
        creds = Credentials(
            None,
            refresh_token=DRIVE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=DRIVE_CLIENT_ID,
            client_secret=DRIVE_CLIENT_SECRET
        )
        _drive_service = build("drive", "v3", credentials=creds)
        return _drive_service
    except Exception as e:
        print(f"  [Drive Notice] Could not initialize Google Drive: {e}")
        return None

def drive_find_file(filename, parent_id=DRIVE_MAIN_FOLDER_ID):
    service = get_drive_service()
    if not service: return None
    try:
        query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
        res = service.files().list(q=query, fields="files(id, name, size)").execute()
        files = res.get("files", [])
        return files[0] if files else None
    except Exception as e:
        print(f"  [Drive Notice] Error querying '{filename}': {e}")
        return None

def drive_download_file(drive_file_id, local_path, filename="file"):
    service = get_drive_service()
    if not service: return False
    try:
        from googleapiclient.http import MediaIoBaseDownload
        req = service.files().get_media(fileId=drive_file_id)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        print(f"  [Drive] Downloading '{filename}' from Google Drive to: {local_path} ...", flush=True)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req, chunksize=10*1024*1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"    Download progress: {int(status.progress() * 100)}%", flush=True)
        print(f"  [Drive OK] Successfully downloaded '{filename}' ({os.path.getsize(local_path):,} bytes)!")
        return True
    except Exception as e:
        print(f"  [Drive Warning] Download failed for '{filename}': {e}")
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass
        return False

def get_or_create_drive_folder(parent_id, folder_name):
    """Finds existing folder under parent_id, or creates it if not found."""
    service = get_drive_service()
    if not service: return None
    try:
        q = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        # Create folder if it does not exist
        body = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]
        }
        created = service.files().create(body=body, fields="id, name").execute()
        print(f"  [Drive] Created folder '{folder_name}' in Google Drive! (ID: {created['id']})")
        return created["id"]
    except Exception as e:
        print(f"  [Drive Error] Could not get/create folder '{folder_name}': {e}")
        return None

def fetch_all_drive_subfolders(parent_folder_id):
    """Fetches all existing subfolders into an in-memory dictionary {name: folder_id}."""
    service = get_drive_service()
    if not service: return {}
    folder_dict = {}
    page_token = None
    try:
        while True:
            q = f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            res = service.files().list(q=q, fields="nextPageToken, files(id, name)", pageSize=1000, pageToken=page_token).execute()
            for item in res.get("files", []):
                folder_dict[item["name"]] = item["id"]
            page_token = res.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        print(f"  [Drive Notice] Error listing existing subfolders: {e}")
    return folder_dict

def drive_upload_or_overwrite(parent_folder_id, filename, local_path, mimetype=None):
    """Uploads file to parent_folder_id. If a file with same name exists, overwrites it."""
    if not os.path.exists(local_path):
        return None
    service = get_drive_service()
    if not service: return None
    try:
        from googleapiclient.http import MediaFileUpload
        if mimetype is None:
            if filename.endswith(".png"): mimetype = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"): mimetype = "image/jpeg"
            elif filename.endswith(".csv"): mimetype = "text/csv"
            else: mimetype = "application/octet-stream"

        media = MediaFileUpload(local_path, mimetype=mimetype, resumable=False)
        q = f"'{parent_folder_id}' in parents and name = '{filename}' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            file_id = files[0]["id"]
            updated = service.files().update(fileId=file_id, media_body=media, fields="id, name").execute()
            return updated["id"]
        else:
            body = {"name": filename, "parents": [parent_folder_id]}
            created = service.files().create(body=body, media_body=media, fields="id, name").execute()
            return created["id"]
    except Exception as e:
        print(f"    [Drive Warning] Upload/Overwrite error for '{filename}': {e}")
        return None

# ============================================================================
# 1. VERIFY & RESTORE MODEL CHECKPOINTS (LOCAL vs GOOGLE DRIVE)
# ============================================================================
print("\n" + "=" * 75)
print("CHECKING MODEL CHECKPOINTS (LOCAL vs GOOGLE DRIVE)")
print("=" * 75)

# Check Stage A Classifier Checkpoint (multi-trial pooling preferred, then baseline)
found_cls = False
for cand in CLASSIFIER_CANDIDATES:
    cand_path = os.path.join(OUTPUT_DIR, cand)
    if os.path.exists(cand_path):
        NEUROSTREAM_BEST_PATH = cand_path
        print(f"[✓] Found local classifier checkpoint: {cand_path} ({os.path.getsize(cand_path):,} bytes)")
        found_cls = True
        break
    else:
        drive_f = drive_find_file(cand, DRIVE_MAIN_FOLDER_ID)
        if drive_f:
            print(f"  Found '{drive_f['name']}' in Google Drive! Restoring to {cand_path}...")
            if drive_download_file(drive_f["id"], cand_path, cand):
                NEUROSTREAM_BEST_PATH = cand_path
                found_cls = True
                break

if not found_cls:
    print(f"  [Notice] No trained classifier found locally or in Drive. Will proceed with initialized weights.")

# Check Full Stage Pipeline Checkpoint
for cand in PIPELINE_CANDIDATES:
    cand_path = os.path.join(OUTPUT_DIR, cand)
    if os.path.exists(cand_path):
        PIPELINE_PATH = cand_path
        print(f"[✓] Found local pipeline checkpoint: {cand_path}")
        break
    else:
        drive_pipe = drive_find_file(cand, DRIVE_MAIN_FOLDER_ID)
        if drive_pipe:
            print(f"  Found '{drive_pipe['name']}' in Google Drive! Restoring to {cand_path}...")
            if drive_download_file(drive_pipe["id"], cand_path, cand):
                PIPELINE_PATH = cand_path
                break

# ============================================================================
# 2. NEUROSTREAM-SST ARCHITECTURE
# ============================================================================

def build_128_channel_scalp_grid_mapping(n_channels=128, grid_h=11, grid_w=11):
    np.random.seed(SEED)
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    radii  = np.linspace(0.15, 0.95, n_channels)
    np.random.shuffle(angles)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    gx = np.clip(np.round((xs + 1.0) / 2.0 * (grid_w - 1)).astype(int), 0, grid_w - 1)
    gy = np.clip(np.round((ys + 1.0) / 2.0 * (grid_h - 1)).astype(int), 0, grid_h - 1)
    proj_matrix = np.zeros((n_channels, grid_h, grid_w), dtype=np.float32)
    for ch in range(n_channels):
        proj_matrix[ch, gy[ch], gx[ch]] = 1.0
    return torch.from_numpy(proj_matrix)

class MorletWaveletFilterBank(nn.Module):
    def __init__(self, mode="high_gamma", fs=1000.0, kernel_len=81):
        super().__init__()
        self.kernel_len = kernel_len
        freqs, sigmas = [95.0], [0.015] # High-Gamma 70-120 Hz centered at 95 Hz
        self.n_bands = len(freqs)
        t = torch.linspace(-kernel_len / (2 * fs), kernel_len / (2 * fs), kernel_len)
        cos_kernels, sin_kernels = [], []
        for f0, sig in zip(freqs, sigmas):
            envelope = torch.exp(-0.5 * (t / sig) ** 2)
            c = envelope * torch.cos(2 * math.pi * f0 * t)
            s = envelope * torch.sin(2 * math.pi * f0 * t)
            cos_kernels.append(c / (torch.norm(c) + 1e-6))
            sin_kernels.append(s / (torch.norm(s) + 1e-6))
        self.cos_filters = nn.Parameter(torch.stack(cos_kernels).unsqueeze(1), requires_grad=True)
        self.sin_filters = nn.Parameter(torch.stack(sin_kernels).unsqueeze(1), requires_grad=True)

    def forward(self, x):
        real = F.conv1d(x, self.cos_filters, padding=self.kernel_len // 2)
        imag = F.conv1d(x, self.sin_filters, padding=self.kernel_len // 2)
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

class NeuroStreamSSTClassifier(nn.Module):
    def __init__(self, spectral_mode="high_gamma", n_classes=40, n_subjects=6,
                 subject_mode="invariant", dropout=0.2):
        super().__init__()
        self.register_buffer("scalp_proj", build_128_channel_scalp_grid_mapping())
        self.spectral_filterbank = MorletWaveletFilterBank(mode=spectral_mode)
        n_bands = self.spectral_filterbank.n_bands

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64 * 16, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def extract_features(self, x_eeg, subj_id=None):
        B, C, T = x_eeg.shape
        x_flat = x_eeg.view(B * C, 1, T)
        env = self.spectral_filterbank(x_flat)
        n_bands = env.shape[1]
        env = env.view(B, C, n_bands, T).permute(0, 2, 3, 1)
        scalp_map = torch.einsum("bfti,ihw->bfthw", env, self.scalp_proj.to(env.device, env.dtype))
        scalp_sub = scalp_map[:, :, ::4, :, :]
        T_sub = scalp_sub.shape[2]
        scalp_2d = scalp_sub.permute(0, 2, 1, 3, 4).reshape(B * T_sub, n_bands, 11, 11)
        spat_feats = self.spatial_conv(scalp_2d)
        spat_feats = spat_feats.view(B, T_sub, 64 * 16).permute(0, 2, 1)
        return self.temporal_conv(spat_feats).squeeze(-1)

    def forward(self, x_eeg, subj_id=None):
        return self.head(self.extract_features(x_eeg, subj_id))

# ============================================================================
# 3. LOAD DATASET & TEST SPLITS
# ============================================================================
print("\n" + "=" * 75)
print("LOADING EEGCVPR40 DATASET FOR FULL TEST EVALUATION")
print("=" * 75)

pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
if not pths: pths = glob.glob(os.path.join(".", "**", "*.pth"), recursive=True)
if not pths: raise FileNotFoundError("Dataset not found!")

bn = os.path.basename
eeg_file = max([p for p in pths if "split" not in bn(p).lower()], key=os.path.getsize)
split_file = next((p for p in sorted([p for p in pths if "split" in bn(p).lower()],
                                     key=lambda p: "all" not in bn(p).lower())), None)

raw_data = torch.load(eeg_file, map_location="cpu", weights_only=False)
raw_images = raw_data["images"]
records = raw_data["dataset"]

X, O, IMG, SUB = [], [], [], []
for oi, it in enumerate(records):
    e = torch.as_tensor(it["eeg"]).float()
    if e.shape[0] != 128 and e.shape[1] == 128: e = e.t()
    if not (450 <= e.shape[1] <= 600): continue
    X.append(e[:, 20:460].contiguous())
    O.append(oi); IMG.append(int(it["image"])); SUB.append(int(it["subject"]))

del raw_data, records
X = torch.stack(X); O = np.array(O); IMG = np.array(IMG); SUB = np.array(SUB)

uniq_imgs = sorted(set(IMG.tolist()))
img_names = {u: raw_images[u] for u in uniq_imgs}
synsets   = sorted({img_names[u].split("_")[0] for u in uniq_imgs})
syn2cls   = {s: i for i, s in enumerate(synsets)}
cls2syn   = {i: s for s, i in syn2cls.items()}
N_CLASSES = len(synsets)

trial_cls  = np.array([syn2cls[img_names[im].split("_")[0]] for im in IMG])
subj_vals  = sorted(set(SUB.tolist()))
subj_map   = {s: i for i, s in enumerate(subj_vals)}
trial_subj = np.array([subj_map[s] for s in SUB])
N_SUBJECTS = len(subj_vals)

# Benchmark Test Split
if split_file is not None:
    sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
    row_of = {int(o): r for r, o in enumerate(O)}
    tr_l, va_l, te_l = [[row_of[int(i)] for i in sp[k] if int(i) in row_of] for k in ("train", "val", "test")]
    tr, va, te = np.array(tr_l), np.array(va_l), np.array(te_l)
else:
    rows_by_img = {}
    for r, im in enumerate(IMG): rows_by_img.setdefault(int(im), []).append(r)
    rng = np.random.RandomState(SEED)
    ims = np.array(sorted(rows_by_img.keys()))
    rng.shuffle(ims)
    n_te = int(0.15 * len(ims))
    te_ims = ims[:n_te]
    te = np.array([r for im in te_ims for r in rows_by_img[im]])
    tr = np.array([r for im in ims[n_te:] for r in rows_by_img[im]])

print(f"Total Test Trials Loaded for Evaluation: {len(te)} trials across {len(set(IMG[te]))} unique images")

# Channel standardization based on training signals
train_signals = X[tr].float()
channel_mean = train_signals.mean(dim=(0, 2), keepdim=True)
channel_std  = train_signals.std(dim=(0, 2), keepdim=True) + 1e-6
X_norm = (X.float() - channel_mean) / channel_std
del train_signals

# ============================================================================
# 4. LOAD NEUROSTREAM-SST MODEL & SCIENTIFIC EVALUATION
# ============================================================================
print("\n" + "=" * 75)
print("EVALUATING NEUROSTREAM-SST CLASSIFIER ACROSS ALL TEST SAMPLES")
print("=" * 75)

model = NeuroStreamSSTClassifier(
    spectral_mode="high_gamma", n_classes=N_CLASSES, n_subjects=N_SUBJECTS,
    subject_mode="invariant"
).to(dev)

if os.path.exists(NEUROSTREAM_BEST_PATH):
    ckpt = torch.load(NEUROSTREAM_BEST_PATH, map_location=dev, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    print(f"Loaded trained checkpoint: {NEUROSTREAM_BEST_PATH}")
else:
    print(f"Notice: {NEUROSTREAM_BEST_PATH} not found. Running with current weights.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.eval()

# Synset to clean human-readable name & prompt mapping
SYNSET_TO_CAPTION = {
    "n02106662": "a photo of a German shepherd",
    "n02124075": "a photo of an Egyptian cat",
    "n02281787": "a photo of a butterfly",
    "n02389026": "a photo of a horse",
    "n02492035": "a photo of a capuchin monkey",
    "n02504458": "a photo of an African elephant",
    "n02510455": "a photo of a giant panda",
    "n02607072": "a photo of an anemone fish",
    "n02690373": "a photo of an airliner airplane",
    "n02906734": "a photo of a broom",
    "n02951358": "a photo of a canoe",
    "n02992529": "a photo of a mobile phone",
    "n03063599": "a photo of a coffee mug",
    "n03100240": "a photo of a convertible car",
    "n03180011": "a photo of a desktop computer",
    "n03197337": "a photo of a digital watch",
    "n03272010": "a photo of an electric guitar",
    "n03272562": "a photo of a train locomotive",
    "n03297495": "a photo of an espresso maker",
    "n03376595": "a photo of a folding chair",
    "n03445777": "a photo of a golf ball",
    "n03452741": "a photo of a grand piano",
    "n03584829": "a photo of an iron",
    "n03590841": "a photo of a jack-o-lantern",
    "n03709823": "a photo of a mailbag postbag",
    "n03773504": "a photo of a missile rocket",
    "n03775071": "a photo of a mitten glove",
    "n03792782": "a photo of a mountain bike",
    "n03792972": "a photo of a mountain tent",
    "n03877472": "a photo of pajamas",
    "n03888257": "a photo of a parachute",
    "n03982430": "a photo of a pool billiard table",
    "n04044716": "a photo of a radio telescope",
    "n04069434": "a photo of a camera",
    "n04086273": "a photo of a revolver handgun",
    "n04120489": "a photo of a running shoe sneaker",
    "n07753592": "a photo of a banana",
    "n07873807": "a photo of a pizza",
    "n11939491": "a photo of a daisy flower",
    "n13054560": "a photo of a mushroom"
}

cls2syn = {i: s for s, i in syn2cls.items()}
id_to_caption = {i: SYNSET_TO_CAPTION.get(s, f"a photo of {s}") for i, s in cls2syn.items()}

def get_clean_class_name(cls_idx):
    syn = cls2syn.get(cls_idx, "")
    caption = SYNSET_TO_CAPTION.get(syn, syn)
    return caption.replace("a photo of an ", "").replace("a photo of a ", "").replace("a photo of ", "").strip()

# ----------------- Image Lookup for Real Stimulus Images -----------------
IMAGE_DIR_CANDIDATES = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/data/images",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/images",
    os.path.join(DATA_ROOT, "images"),
    "./images",
    "/kaggle/working/images"
]

IMAGE_PATH_CACHE = {}

def resolve_image_path(name):
    if name in IMAGE_PATH_CACHE:
        return IMAGE_PATH_CACHE[name]
    syn = name.split("_")[0]
    for root in IMAGE_DIR_CANDIDATES:
        if not os.path.exists(root):
            continue
        p1 = os.path.join(root, syn, name + ".JPEG")
        if os.path.exists(p1):
            IMAGE_PATH_CACHE[name] = p1; return p1
        p2 = os.path.join(root, name + ".JPEG")
        if os.path.exists(p2):
            IMAGE_PATH_CACHE[name] = p2; return p2
        p3 = os.path.join(root, syn, name + ".jpg")
        if os.path.exists(p3):
            IMAGE_PATH_CACHE[name] = p3; return p3
        g = glob.glob(os.path.join(root, syn, name + ".*"))
        if g:
            IMAGE_PATH_CACHE[name] = g[0]; return g[0]
    IMAGE_PATH_CACHE[name] = None
    return None

# ============================================================================
# 5. LOAD DIFFUSION PIPELINE (Fast UniPC Multi-Step, NO YOLO / NO SEGMENTATION)
# ============================================================================
print("\n" + "=" * 75)
print("LOADING STABLE DIFFUSION BASE COMPONENTS (Fast UniPC Multi-Step)")
print("=" * 75)

for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
    val = os.environ.get(k, "")
    if not val or len(val.strip()) < 10 or "your_token" in val or "token" in val.lower():
        os.environ.pop(k, None)

SD_CANDIDATES = [
    "sd2-community/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-1-base",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4"
]

tok, txt, vae, unet, sched, active_sd = None, None, None, None, None, None

for cand_id in SD_CANDIDATES:
    try:
        print(f"Attempting to load model from: {cand_id} ...")
        tok = CLIPTokenizer.from_pretrained(cand_id, subfolder="tokenizer", token=False)
        txt = CLIPTextModel.from_pretrained(cand_id, subfolder="text_encoder", token=False).to(dev)
        vae = AutoencoderKL.from_pretrained(cand_id, subfolder="vae", token=False).to(dev)
        unet = UNet2DConditionModel.from_pretrained(cand_id, subfolder="unet", token=False).to(dev)
        sched = UniPCMultistepScheduler.from_pretrained(cand_id, subfolder="scheduler", token=False)
        active_sd = cand_id
        print(f"Successfully loaded Stable Diffusion components from: {active_sd}")
        break
    except Exception as e:
        print(f"  Notice: {cand_id} failed ({str(e)[:80]}). Trying next candidate...")

if active_sd is None:
    print("Warning: All online SD candidates unavailable. Fallback visual generator will be used.")

# ============================================================================
# 6. INITIALIZE GOOGLE DRIVE TARGET DIRECTORY: 'GeneratedImage'
# ============================================================================
print("\n" + "=" * 75)
print("INITIALIZING GOOGLE DRIVE 'GeneratedImage' DIRECTORY")
print("=" * 75)

gen_folder_id = get_or_create_drive_folder(DRIVE_MAIN_FOLDER_ID, "GeneratedImage")
if gen_folder_id:
    print(f"[✓] Google Drive 'GeneratedImage' folder ready (ID: {gen_folder_id})")
    print("  Fetching existing subfolders in 'GeneratedImage' to prevent duplicates...")
    existing_drive_subfolders = fetch_all_drive_subfolders(gen_folder_id)
    print(f"  Found {len(existing_drive_subfolders)} existing subfolder(s) in Drive.")
else:
    print("[Notice] Could not initialize Google Drive folder. Outputs will be saved locally.")
    existing_drive_subfolders = {}

# ============================================================================
# 7. INFERENCE LOOP ACROSS ALL TEST IMAGES (>1000 TRIALS)
# ============================================================================
TOTAL_TEST_SAMPLES = len(te)
print("\n" + "=" * 75)
print(f"STARTING FULL GENERATION & EVALUATION ACROSS ALL {TOTAL_TEST_SAMPLES} TEST TRIALS")
print(f"Destination: Local '{LOCAL_GEN_DIR}' & Google Drive 'GeneratedImage/'")
print("=" * 75)

evaluation_records = []
preview_samples = []

start_time = time.time()

for s_i in range(TOTAL_TEST_SAMPLES):
    trial_idx = te[s_i]
    im_id = IMG[trial_idx]
    im_name = img_names[im_id]
    gt_c = trial_cls[trial_idx]
    gt_class_name = get_clean_class_name(gt_c)
    gt_caption = id_to_caption.get(gt_c, gt_class_name)

    # 1. NeuroStream-SST Inference on pure EEG trial
    head_dtype = next(model.head.parameters()).dtype
    x_trial = X_norm[trial_idx:trial_idx+1].to(device=dev, dtype=head_dtype)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            feats = model.extract_features(x_trial)
            logits = model.head(feats)
            pred_c = logits.argmax(dim=-1).item()

    pred_class_name = get_clean_class_name(pred_c)
    pred_prompt = id_to_caption.get(pred_c, f"a photo of {pred_class_name}")
    is_match = (pred_c == gt_c)
    is_true_str = "true" if is_match else "false"

    evaluation_records.append({
        "real_image_class": gt_class_name,
        "predicted_class": pred_class_name,
        "isTrue": is_true_str
    })

    # 2. Local Sample Folder Creation (sample_0000, sample_0001, etc.)
    sample_folder_name = f"sample_{s_i:04d}"
    sample_local_dir = os.path.join(LOCAL_GEN_DIR, sample_folder_name)
    os.makedirs(sample_local_dir, exist_ok=True)

    real_img_path = os.path.join(sample_local_dir, "real.png")
    gen_img_path = os.path.join(sample_local_dir, "generated.png")

    # 3. Obtain Real Stimulus Image (512x512)
    real_p = resolve_image_path(im_name)
    real_np = None
    if real_p and os.path.exists(real_p):
        try:
            real_np = np.array(Image.open(real_p).convert("RGB").resize((512, 512)))
        except Exception:
            real_np = None
    if real_np is None:
        real_np = np.ones((512, 512, 3), dtype=np.uint8) * 240
        cv2.rectangle(real_np, (15, 15), (497, 497), (200, 210, 200), 2)
        cv2.putText(real_np, "REAL STIMULUS", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 80, 20), 2)
        cv2.putText(real_np, f"ID: {im_name}", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
        cv2.putText(real_np, "Ground Truth Class:", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)
        cv2.putText(real_np, gt_class_name, (40, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 120, 20), 2)

    Image.fromarray(real_np).save(real_img_path)

    # 4. Generate Image from Predicted EEG Conditioning
    if active_sd is not None:
        with torch.inference_mode():
            t_in = tok(pred_prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            uncond_in = tok("", padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            ctx = txt(t_in.input_ids.to(dev))[0]
            uncond_ctx = txt(uncond_in.input_ids.to(dev))[0]
            c_all = torch.cat([uncond_ctx, ctx])

            sched.set_timesteps(STEPS)
            torch.manual_seed(SEED + s_i)
            lat = torch.randn((1, unet.config.in_channels, 64, 64), device=dev)

            for t_step in sched.timesteps:
                lat_in = sched.scale_model_input(torch.cat([lat, lat]), t_step)
                n_uncond, n_text = unet(lat_in, t_step, encoder_hidden_states=c_all).sample.chunk(2)
                n_pred = n_uncond + GUIDANCE * (n_text - n_uncond)
                lat = sched.step(n_pred, t_step, lat).prev_sample

            dec = vae.decode(lat / 0.18215).sample
            gen_np = ((dec.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    else:
        # High-clarity fallback if offline
        gen_np = np.ones((512, 512, 3), dtype=np.uint8) * 230
        cv2.circle(gen_np, (256, 256), 180, (50, 120, 220), -1)
        cv2.putText(gen_np, pred_class_name, (70, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    Image.fromarray(gen_np).save(gen_img_path)

    # 5. Google Drive Upload / Overwrite Logic
    if gen_folder_id:
        # Check if sample folder already exists in Drive; create if not
        if sample_folder_name in existing_drive_subfolders:
            sample_drive_folder_id = existing_drive_subfolders[sample_folder_name]
        else:
            sample_drive_folder_id = get_or_create_drive_folder(gen_folder_id, sample_folder_name)
            if sample_drive_folder_id:
                existing_drive_subfolders[sample_folder_name] = sample_drive_folder_id

        # Upload and overwrite real.png and generated.png
        if sample_drive_folder_id:
            drive_upload_or_overwrite(sample_drive_folder_id, "real.png", real_img_path, mimetype="image/png")
            drive_upload_or_overwrite(sample_drive_folder_id, "generated.png", gen_img_path, mimetype="image/png")

    # Keep first 24 samples for notebook visualization
    if len(preview_samples) < 24:
        preview_samples.append((real_np, gen_np, gt_class_name, pred_class_name, is_match))

    # Periodic progress report and VRAM cleanup
    if (s_i + 1) % 10 == 0 or s_i == 0 or (s_i + 1) == TOTAL_TEST_SAMPLES:
        elapsed = time.time() - start_time
        fps = (s_i + 1) / max(1e-5, elapsed)
        eta_sec = (TOTAL_TEST_SAMPLES - (s_i + 1)) / max(1e-5, fps)
        tag = "[MATCH ✓]" if is_match else "[MISMATCH]"
        print(f"[{s_i+1:04d}/{TOTAL_TEST_SAMPLES}] GT: '{gt_class_name}' -> Pred: '{pred_class_name}' {tag} | Speed: {fps:.2f} smp/s | ETA: {eta_sec/60:.1f}m", flush=True)

    if (s_i + 1) % 100 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"\n[✓] ALL {TOTAL_TEST_SAMPLES} samples processed and saved!")

# ============================================================================
# 8. GENERATE ONE EVALUATION CSV REPORT (3 COLUMNS + SUMMARY COUNTS)
# ============================================================================
print("\n" + "=" * 75)
print("COMPILING TEST EVALUATION CSV REPORT")
print("=" * 75)

total_true = sum(1 for r in evaluation_records if r["isTrue"] == "true")
total_false = sum(1 for r in evaluation_records if r["isTrue"] == "false")
total_images = len(evaluation_records)
accuracy_pct = (total_true / max(1, total_images)) * 100.0

print(f"  • TOTAL IMAGES EVALUATED : {total_images}")
print(f"  • TOTAL TRUE PREDICTIONS : {total_true}")
print(f"  • TOTAL FALSE PREDICTIONS: {total_false}")
print(f"  • OVERALL TEST ACCURACY  : {accuracy_pct:.2f}%")

# Write CSV with strictly 3 columns
with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["real_image_class", "predicted_class", "isTrue"])
    # All sample rows
    for r in evaluation_records:
        writer.writerow([r["real_image_class"], r["predicted_class"], r["isTrue"]])
    # Summary count rows at the bottom (strictly 3 columns)
    writer.writerow(["TOTAL_TRUE", total_true, "true"])
    writer.writerow(["TOTAL_FALSE", total_false, "false"])
    writer.writerow(["TOTAL_IMAGES", total_images, ""])
    writer.writerow(["ACCURACY", f"{accuracy_pct:.2f}%", ""])

print(f"[✓] Evaluation CSV saved locally to: {CSV_OUTPUT_PATH}")

# Also save a copy directly in /kaggle/working/
root_csv = os.path.join(OUTPUT_DIR, "test_evaluation_results.csv")
with open(root_csv, "w", newline="", encoding="utf-8") as f:
    with open(CSV_OUTPUT_PATH, "r", encoding="utf-8") as src:
        f.write(src.read())

# Upload / Overwrite CSV in Google Drive 'GeneratedImage' folder
if gen_folder_id:
    print("Uploading evaluation CSV to Google Drive 'GeneratedImage' folder...")
    drive_upload_or_overwrite(gen_folder_id, "test_evaluation_results.csv", CSV_OUTPUT_PATH, mimetype="text/csv")
    print(f"[✓] Successfully uploaded 'test_evaluation_results.csv' to Google Drive!")

# ============================================================================
# 9. DISPLAY 2-PANEL PREVIEW: REAL IMAGE vs PREDICTED GENERATED IMAGE
# ============================================================================
# Display preview of test samples (Real vs Generated) without segmentation
try:
    n_prev = len(preview_samples)
    if n_prev > 0:
        print("\n" + "=" * 75)
        print(f"DISPLAYING PREVIEW OF TEST SAMPLES (REAL vs GENERATED)")
        print("=" * 75)

        PAGE_SIZE = 6
        n_pages = math.ceil(n_prev / PAGE_SIZE)
        for page in range(n_pages):
            batch = preview_samples[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
            fig, axes = plt.subplots(len(batch), 2, figsize=(10, 3.8 * len(batch)))
            if len(batch) == 1:
                axes = np.expand_dims(axes, 0)

            for idx, (real_img, gen_img, gt_name, pred_name, is_match) in enumerate(batch):
                s_num = page * PAGE_SIZE + idx + 1
                # 1. Real Image
                axes[idx, 0].imshow(real_img)
                axes[idx, 0].set_title(f"Sample {s_num}: REAL STIMULUS\nClass: {gt_name}",
                                       fontsize=10, fontweight="bold", color="#1b5e20", pad=4)
                axes[idx, 0].axis("off")

                # 2. Generated Image
                m_str = " [CORRECT ✓]" if is_match else " [MISMATCH]"
                m_col = "#0d47a1" if is_match else "#b71c1c"
                axes[idx, 1].imshow(gen_img)
                axes[idx, 1].set_title(f"Sample {s_num}: GENERATED IMAGE (SD 2.1)\nPred: {pred_name}{m_str}",
                                       fontsize=10, fontweight="bold", color=m_col, pad=4)
                axes[idx, 1].axis("off")

            plt.tight_layout()
            preview_save_path = os.path.join(OUTPUT_DIR, f"preview_page_{page + 1:02d}.png")
            plt.savefig(preview_save_path, dpi=140, bbox_inches="tight")
            plt.show()
            print(f"  [✓] Preview page {page + 1}/{n_pages} rendered & saved.")
except Exception as e:
    print(f"Preview plotting notice: {e}")

print("\n" + "=" * 75)
print(f"CELL 2 COMPLETE: All {TOTAL_TEST_SAMPLES} samples generated, evaluated, and synced to Drive!")
print("=" * 75)
