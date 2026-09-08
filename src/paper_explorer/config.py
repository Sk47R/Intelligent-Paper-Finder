from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
PLOTS_DIR = DATA_DIR / "plots"
DEFAULT_STORE_PATH = PROCESSED_DIR / "papers.json"
DEFAULT_INDEX_PATH = INDEX_DIR / "papers.faiss"
DEFAULT_ID_MAP_PATH = INDEX_DIR / "id_map.json"
DEFAULT_PLOTS_DIR = PLOTS_DIR