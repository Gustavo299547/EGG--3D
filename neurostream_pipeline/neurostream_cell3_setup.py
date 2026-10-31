# =====================================================================
# NEUROSTREAM CELL 3 — COMPLETE TRELLIS SETUP (idempotent; safe to re-run)
# Requirements: Kaggle Internet ON, GPU T4 x2 (or any T4)
# =====================================================================
import os, sys, subprocess, shutil, time, textwrap

ROOT  = "/kaggle/working"
VENV  = f"{ROOT}/trellis_env"
PY    = f"{VENV}/bin/python"
REPO  = f"{ROOT}/TRELLIS"
EXT   = f"{ROOT}/extensions"
CUDA  = "/usr/local/cuda-11.8"
HF    = f"{ROOT}/hf_cache"
CONSTRAINTS = f"{ROOT}/constraints.txt"
os.makedirs(EXT, exist_ok=True)

NVCC_FLAGS = "-allow-unsupported-compiler"
ENV = os.environ.copy()
ENV.update({
    "CUDA_HOME": CUDA,
    "CUDA_PATH": CUDA,
    "PATH": f"{VENV}/bin:{CUDA}/bin:" + ENV.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA}/lib64:" + ENV.get("LD_LIBRARY_PATH", ""),
    "TORCH_CUDA_ARCH_LIST": "7.5",
    "MAX_JOBS": "2",
    "FORCE_CUDA": "1",
    "NVCC_APPEND_FLAGS": NVCC_FLAGS,
    "CUDAFLAGS": NVCC_FLAGS,
    "TORCH_NVCC_FLAGS": NVCC_FLAGS,
    "ATTN_BACKEND": "xformers",
    "SPCONV_ALGO": "native",
    "HF_HOME": HF,
    "PYTHONPATH": REPO,
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
})

# Install gcc-11/g++-11 and OpenGL/EGL headers if needed (CUDA 11.8 compatible host compiler)
if not os.path.exists("/usr/bin/gcc-11"):
    subprocess.run("apt-get update -qq && apt-get install -y -qq gcc-11 g++-11 build-essential pkg-config libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev",
                   shell=True, env=ENV)

if os.path.exists("/usr/bin/gcc-11") and os.path.exists("/usr/bin/g++-11"):
    ENV["CC"] = "/usr/bin/gcc-11"
    ENV["CXX"] = "/usr/bin/g++-11"
    ENV["CUDAHOSTCXX"] = "/usr/bin/g++-11"


def tail(s, n=150):
    lines = (s or "").strip().splitlines()
    return "\n".join(lines[-n:])

def sh(cmd, desc="", cwd=None, check=True):
    shell = isinstance(cmd, str)
    t0 = time.time()
    if desc: print(f"\n>>> {desc}", flush=True)
    p = subprocess.run(cmd, shell=shell, cwd=cwd, env=ENV,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 and check:
        print("\n" + "!" * 78)
        print(f"COMMAND FAILED: {cmd if shell else ' '.join(cmd)}")
        print(f"exit code: {p.returncode}  (after {time.time()-t0:.1f}s)")
        print("--- STDOUT (tail) ---")
        print(tail(p.stdout))
        print("--- STDERR (tail) ---")
        print(tail(p.stderr))
        print("!" * 78)
        raise RuntimeError(f"Step failed: {desc or cmd}")
    if desc: print(f"    done in {time.time()-t0:.1f}s", flush=True)
    return p

# 1. Base venv (Auto-bootstraps Python 3.10 via uv if python3.10 is not in system PATH)
if not os.path.exists(f"{VENV}/bin/activate"):
    if shutil.which("python3.10"):
        sh(f"python3.10 -m venv {VENV}", "Create Python 3.10 virtualenv")
    else:
        print("Notice: 'python3.10' not found in PATH. Using uv to fetch official Python 3.10...")
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], "Install uv")
        sh(["uv", "venv", "--python", "3.10", "--seed", VENV], "Create Python 3.10 virtualenv via uv")

# Pin setuptools < 70
with open(CONSTRAINTS, "w") as f:
    f.write("setuptools<70.0.0\nwheel\npip\n")

sh(f"{PY} -m pip install -q -c {CONSTRAINTS} 'setuptools<70.0.0' wheel", "Bootstrap pip/setuptools")

# 2. PyTorch 2.4.0 + cu118 + torchvision + xformers
chk_torch = sh(f"{PY} -c \"import torch; assert torch.__version__.startswith('2.4.0') and 'cu118' in torch.__version__\"", check=False)
if chk_torch.returncode != 0:
    sh(f"{PY} -m pip install -q -c {CONSTRAINTS} "
       "torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118",
       "Install PyTorch 2.4.0+cu118")

chk_xf = sh(f"{PY} -c \"import xformers; assert xformers.__version__.startswith('0.0.27.post2')\"", check=False)
if chk_xf.returncode != 0:
    sh(f"{PY} -m pip install -q -c {CONSTRAINTS} "
       "xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118",
       "Install xformers 0.0.27.post2 (official cu118 wheel)")

# 3. Clone TRELLIS repo
if not os.path.exists(f"{REPO}/.git"):
    sh(f"git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git {REPO}", "Clone TRELLIS")

# 4. Core pip requirements
reqs = [
    "trimesh", "pillow", "scipy", "timm", "rembg", "imageio", "imageio-ffmpeg",
    "einops", "easydict", "huggingface_hub", "safetensors", "pyglet",
    "accelerate", "transformers==4.46.3", "plyfile", "onnxruntime", "ninja",
    "open3d",
    "git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d",
]
sh(f"{PY} -m pip install -q -c {CONSTRAINTS} " + " ".join(f"'{r}'" for r in reqs), "Install python requirements")

# 5. Pre-built spconv & kaolin wheels
chk_sp = sh(f"{PY} -c \"import spconv\"", check=False)
if chk_sp.returncode != 0:
    sh(f"{PY} -m pip install -q spconv-cu118", "Install spconv-cu118 wheel")

chk_k = sh(f"{PY} -c \"import kaolin\"", check=False)
if chk_k.returncode != 0:
    sh(f"{PY} -m pip install -q kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html", "Install kaolin wheel", check=False)

# 6. Build C++/CUDA extensions from TRELLIS submodules
def build_submodule(rel_dir, desc, install_name=None):
    pkg_dir = os.path.abspath(f"{REPO}/{rel_dir}")
    if install_name:
        chk = sh(f"{PY} -c \"import importlib.util as u; exit(0 if u.find_spec('{install_name}') else 1)\"", check=False)
        if chk.returncode == 0:
            print(f">>> {desc} (already installed)")
            return
    if os.path.exists(f"{pkg_dir}/setup.py"):
        sh(f"{PY} -m pip install -q --no-build-isolation -c {CONSTRAINTS} -e {pkg_dir}", f"Build/Install {desc}")

build_submodule("submodules/diff-gaussian-rasterization", "diff-gaussian-rasterization", "diff_gaussian_rasterization")
build_submodule("submodules/spatial-fairing", "spatial-fairing", "spatial_fairing")
build_submodule("submodules/nvdiffrast", "nvdiffrast", "nvdiffrast")

# 7. Download weights
dl_script = textwrap.dedent("""
from huggingface_hub import hf_hub_download
import json, os

repo = "microsoft/TRELLIS-image-large"
hf_hub_download(repo, "pipeline.json")
with open(hf_hub_download(repo, "pipeline.json")) as f:
    cfg = json.load(f)

skip = {"slat_decoder_rf", "slat_decoder_gs"}
for name, rel in cfg["args"]["models"].items():
    if name in skip:
        continue
    for ext in (".json", ".safetensors"):
        print(f"fetching {rel}{ext}")
        hf_hub_download(repo, rel + ext)
print("ALL NEEDED WEIGHTS READY.")
""")

sh([PY, "-c", dl_script], "Prefetch TRELLIS-image-large weights (Mesh-only pipeline)")

print("\n" + "=" * 60)
print("TRELLIS SETUP COMPLETED SUCCESSFULLY!")
print("=" * 60)
