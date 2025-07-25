from pathlib import Path

try:
    import tomllib  # Python ≥3.11
except ModuleNotFoundError:  # pragma: no cover – fallback for older Pythons
    import tomli as tomllib  # type: ignore

with open("settings.toml", "rb") as f:
    CFG = tomllib.load(f)

ROOT_DIR = Path(CFG["paths"]["root_dir"]).expanduser().resolve()
DB_PATH = Path(CFG["paths"]["db_path"]).expanduser().resolve()
MODEL_NAME = CFG["model"]["name"]
EMBED_DIM = int(CFG["model"]["embed_dim"])

CHUNK_LEN = int(CFG["index"]["chunk_len"])
CHUNK_OVERLAP = int(CFG["index"]["chunk_overlap"])
TOP_K = int(CFG["index"]["top_k"])

# How many candidates to pull from each modality before fusion
POOL_SIZE = int(CFG["index"].get("pool_size", 50))
RRF_K = int(CFG["index"].get("rrf_k", 60))

MAX_FILE_SIZE = int(float(CFG["filters"].get("max_file_size_mb", 5)) * 1024 * 1024)
BLACKLIST_EXT = {ext.lower() for ext in CFG["filters"].get("blacklist_ext", [])}
BLACKLIST_DIRS = set(CFG["filters"].get("blacklist_dirs", []))