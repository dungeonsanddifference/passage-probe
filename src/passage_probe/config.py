from pathlib import Path

try:
    import tomllib  # Python ≥3.11
except ModuleNotFoundError:  # pragma: no cover – fallback for older versions
    import tomli as tomllib  # type: ignore

with open("settings.toml", "rb") as f:
    CFG = tomllib.load(f)

# paths
ROOT_DIR = Path(CFG["paths"]["root_dir"]).expanduser().resolve()
DB_PATH = Path(CFG["paths"]["db_path"]).expanduser().resolve()
MODEL_NAME = CFG["model"]["name"]
EMBED_DIM = int(CFG["model"]["embed_dim"])

# index
CHUNK_LEN = int(CFG["index"]["chunk_len"])
CHUNK_OVERLAP = int(CFG["index"]["chunk_overlap"])
TOP_K = int(CFG["index"]["top_k"])
LINE_BY_LINE_EXT = {".csv"} | {
    ext.lower() for ext in CFG["index"].get("line_by_line_ext", [])
}
POOL_SIZE = int(CFG["index"].get("pool_size", 50))
RRF_K = int(CFG["index"].get("rrf_k", 60))

# filters
MAX_FILE_SIZE = int(float(CFG["filters"].get("max_file_size_mb", 5)) * 1024 * 1024)
BLACKLIST_EXT = {ext.lower() for ext in CFG["filters"].get("blacklist_ext", [])}
BLACKLIST_DIRS = set(CFG["filters"].get("blacklist_dirs", []))

__all__ = [
    "ROOT_DIR", "DB_PATH", "MODEL_NAME", "EMBED_DIM", "CHUNK_LEN",
    "CHUNK_OVERLAP", "TOP_K", "LINE_BY_LINE_EXT", "POOL_SIZE", "RRF_K",
    "MAX_FILE_SIZE", "BLACKLIST_EXT", "BLACKLIST_DIRS"
]