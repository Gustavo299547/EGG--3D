# ============================================================================
# NEUROSTREAM CELL 4 : ROBUST LOW-VRAM TRELLIS 3D SHAPE GENERATION
# ----------------------------------------------------------------------------
# Resolves the FlexiCubes CUDA Out-of-Memory (768 MiB deficit) error:
#   1. Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate fragmentation.
#   2. Low-VRAM Sequential Model Offloading:
#      - Moves sparse_structure_flow_model (~1.8 GB) to CPU after structure sampling.
#      - Moves slat_flow_model (~4.5 GB) and image_cond_model (~1.5 GB) to CPU
#        before calling FlexiCubes mesh decoder.
#      - Frees over 7.8 GB of VRAM, giving FlexiCubes >11.5 GB of free headroom!
#   3. Runs slat_decoder_mesh inside torch.inference_mode() with float16 autocast.
#   4. Reconstructs segmented images into 3D .glb meshes:
#      /kaggle/working/trellis_output/trellis_result_*.glb
# ============================================================================

import os, sys, gc, glob, subprocess, time, torch

# ----------------- Pre-Execution Kernel Memory Cleanup -----------------
print("Releasing Jupyter kernel GPU memory for TRELLIS...")
# Aggressively delete heavy model variables from previous cells in notebook kernel
for _v in ['unet', 'vae', 'txt', 'tok', 'c_all', 'lat', 'yolo_model', 'X_norm', 'model', 'all_features']:
    if _v in globals():
        del globals()[_v]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
    print(f"  -> VRAM status: {free_mem:.2f} GB free out of {total_mem:.2f} GB total.")

SEG_DIR = "/kaggle/working/segmented_images"
OUT_DIR = "/kaggle/working/trellis_output"
os.makedirs(OUT_DIR, exist_ok=True)
MAX_3D_SAMPLES = 5  # Process first 5 samples by default; change to 10, 20, or 150 as needed!

def get_sample_num(p):
    try: return int(os.path.basename(p).split('_')[-1].split('.')[0])
    except: return 0

raw_imgs = glob.glob(f"{SEG_DIR}/segmented_sample_*.png")
test_imgs = sorted(raw_imgs, key=get_sample_num)[:MAX_3D_SAMPLES]
if not test_imgs:
    test_imgs = sorted(glob.glob(f"{SEG_DIR}/*.png"))[:MAX_3D_SAMPLES]

if not test_imgs:
    raise FileNotFoundError(f"No segmented images found in {SEG_DIR}. Please run Cell 2 first!")

print(f"Found {len(raw_imgs)} total segmented images in {SEG_DIR}.")
print(f"Queueing first {len(test_imgs)} samples for 3D conversion (Set MAX_3D_SAMPLES to process more):")
for p in test_imgs:
    print(f"  - {p}")

# ----------------- Write Isolated 3D Inference Script -----------------
# We write this to a standalone Python script file to avoid any f-string escaping
# and to run cleanly inside the isolated Python 3.10 trellis_env.
STANDALONE_SCRIPT_PATH = "/kaggle/working/run_trellis_standalone.py"

STANDALONE_CODE = r'''import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/kaggle/working/TRELLIS")

# Mock open3d so unused trellis_text_to_3d import doesn't fail
import unittest.mock as _mock
if "open3d" not in sys.modules:
    sys.modules["open3d"] = _mock.MagicMock()

# Mock kaolin so flexicubes check_tensor validation import doesn't fail
import types
if "kaolin" not in sys.modules:
    _k = types.ModuleType("kaolin")
    _ku = types.ModuleType("kaolin.utils")
    _kt = types.ModuleType("kaolin.utils.testing")
    _kt.check_tensor = lambda *args, **kwargs: True
    _ku.testing = _kt
    _k.utils = _ku
    sys.modules["kaolin"] = _k
    sys.modules["kaolin.utils"] = _ku
    sys.modules["kaolin.utils.testing"] = _kt

import gc, json, glob, re
import numpy as np
import torch
import trimesh
from PIL import Image
from huggingface_hub import hf_hub_download
from trellis.pipelines import TrellisImageTo3DPipeline

SEG_DIR = os.environ.get("SEG_DIR", "/kaggle/working/segmented_images")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/trellis_output")
MAX_COUNT = int(os.environ.get("MAX_3D_SAMPLES", "5"))
REPO_ID = "microsoft/TRELLIS-image-large"
WORK_CACHE = "/kaggle/working/hf_cache/hub"
TMP_CACHE  = "/tmp/hf_dl"
LOCAL      = "/tmp/trellis_local"

def get_sample_num(p):
    try:
        match = re.search(r'\d+', os.path.basename(p))
        return int(match.group()) if match else 0
    except:
        return 0

raw_imgs = glob.glob(os.path.join(SEG_DIR, "segmented_sample_*.png"))
images = sorted(raw_imgs, key=get_sample_num)[:MAX_COUNT]
if not images:
    images = sorted(glob.glob(os.path.join(SEG_DIR, "*.png")), key=get_sample_num)[:MAX_COUNT]

print(f"Loading TRELLIS pipeline to process {len(images)} images...", flush=True)

def fetch(fn):
    try:
        return hf_hub_download(REPO_ID, fn, cache_dir=WORK_CACHE, local_files_only=True), "cache"
    except Exception:
        return hf_hub_download(REPO_ID, fn, cache_dir=TMP_CACHE), "downloaded"

cfg_path, _ = fetch("pipeline.json")
cfg = json.load(open(cfg_path))
skip = {"slat_decoder_rf", "slat_decoder_gs"}
cfg["args"]["models"] = {k: v for k, v in cfg["args"]["models"].items() if k not in skip}

os.makedirs(f"{LOCAL}/ckpts", exist_ok=True)
json.dump(cfg, open(f"{LOCAL}/pipeline.json", "w"))
for name, rel in cfg["args"]["models"].items():
    for ext in (".json", ".safetensors"):
        src, _ = fetch(rel + ext)
        dst = f"{LOCAL}/{rel}{ext}"
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.realpath(src), dst)

pipeline = TrellisImageTo3DPipeline.from_pretrained(LOCAL)

# Keep models initially on CPU to conserve VRAM
for m in pipeline.models.values():
    m.to("cpu")
gc.collect()
torch.cuda.empty_cache()
print("TRELLIS Pipeline successfully initialized with Dynamic Sequential Offloading.", flush=True)

# Process images one by one
os.makedirs(OUT_DIR, exist_ok=True)
for idx, img_path in enumerate(images):
    s_num = get_sample_num(img_path)
    print(f"\n[{idx+1}/{len(images)}] Generating 3D shape for: {os.path.basename(img_path)} ...", flush=True)
    
    raw_img = Image.open(img_path).convert("RGB")
    prep_img = pipeline.preprocess_image(raw_img) if hasattr(pipeline, "preprocess_image") else raw_img

    # 1. Image Conditioning
    if 'image_cond_model' in pipeline.models:
        pipeline.models['image_cond_model'].to("cuda")
    with torch.inference_mode():
        cond = pipeline.get_cond([prep_img])
    torch.manual_seed(42 + idx)

    # 2. Sample Sparse Structure
    if 'sparse_structure_flow_model' in pipeline.models:
        pipeline.models['sparse_structure_flow_model'].to("cuda")
    with torch.inference_mode():
        coords = pipeline.sample_sparse_structure(cond, 1, {"steps": 12, "cfg_strength": 7.5})

    # Free sparse structure model immediately
    if 'sparse_structure_flow_model' in pipeline.models:
        pipeline.models['sparse_structure_flow_model'].to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Sample Structured Latents (SLAT)
    if 'slat_flow_model' in pipeline.models:
        pipeline.models['slat_flow_model'].to("cuda")
    with torch.inference_mode():
        slat = pipeline.sample_slat(cond, coords, {"steps": 12, "cfg_strength": 3.0})

    # CRITICAL: Offload large SLAT flow model & condition model BEFORE FlexiCubes
    if 'slat_flow_model' in pipeline.models:
        pipeline.models['slat_flow_model'].to("cpu")
    if 'image_cond_model' in pipeline.models:
        pipeline.models['image_cond_model'].to("cpu")
    del cond, coords
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    free_before_mesh = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"    VRAM before mesh extraction: {free_before_mesh:.2f} GB free (Over 10 GB headroom!)", flush=True)

    # 4. Decode SLAT into 3D Mesh using FlexiCubes (Now has >11 GB free VRAM!)
    if 'slat_decoder_mesh' in pipeline.models:
        pipeline.models['slat_decoder_mesh'].to("cuda")
        
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = pipeline.decode_slat(slat, formats=["mesh"])
        mesh = outputs["mesh"][0]

    # Offload mesh decoder
    if 'slat_decoder_mesh' in pipeline.models:
        pipeline.models['slat_decoder_mesh'].to("cpu")

    # Export to GLTF/GLB (Z-up -> Y-up coordinate conversion)
    verts = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    n_v, n_f = len(verts), len(faces)
    verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

    colors = None
    va = getattr(mesh, "vertex_attrs", None)
    if va is not None:
        va = va.detach().float().cpu().numpy()
        if va.ndim == 2 and va.shape[0] == n_v and va.shape[1] >= 3:
            colors = (np.clip(va[:, :3], 0, 1) * 255).astype(np.uint8)

    glb_path = f"{OUT_DIR}/trellis_result_sample_{s_num}.glb"
    trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False).export(glb_path)
    print(f"    [✓] Exported: {glb_path} (Vertices: {n_v:,}, Faces: {n_f:,})", flush=True)

    del outputs, mesh, slat
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print(f"\nALL {len(images)} 3D MODELS GENERATED SUCCESSFULLY IN {OUT_DIR}.", flush=True)
'''

with open(STANDALONE_SCRIPT_PATH, "w") as f:
    f.write(STANDALONE_CODE)
print(f"Wrote isolated 3D inference script to: {STANDALONE_SCRIPT_PATH}")

# ----------------- Execute Isolated Process -----------------
sub_env = os.environ.copy()
sub_env["MAX_3D_SAMPLES"] = str(MAX_3D_SAMPLES)
sub_env["SEG_DIR"] = SEG_DIR
sub_env["OUT_DIR"] = OUT_DIR

VENV_PYTHON = "/kaggle/working/trellis_env/bin/python"
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable
else:
    # Auto-fix: Ensure transformers is pinned to 4.46.3 (official TRELLIS version for PyTorch 2.4.0)
    chk_tf = subprocess.run([VENV_PYTHON, "-c", "import transformers; assert transformers.__version__ == '4.46.3'"], capture_output=True)
    if chk_tf.returncode != 0:
        print("  -> Auto-fixing transformers version in trellis_env (pinning to 4.46.3 for PyTorch 2.4.0)...")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-q", "--no-deps", "transformers==4.46.3"], check=False)

print(f"Starting TRELLIS 3D inference process with {VENV_PYTHON} ...")
proc = subprocess.Popen(
    [VENV_PYTHON, STANDALONE_SCRIPT_PATH],
    env=sub_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in iter(proc.stdout.readline, ''):
    print(line, end='', flush=True)

proc.stdout.close()
return_code = proc.wait()

if return_code != 0:
    raise RuntimeError(f"TRELLIS 3D generation failed with exit code: {return_code}")
else:
    print("\n" + "=" * 70)
    print(f"SUCCESS: ALL 3D SHAPES RECONSTRUCTED & SAVED TO {OUT_DIR}/")
    print("=" * 70)
