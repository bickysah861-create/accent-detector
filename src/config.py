"""Central configuration. All paths are absolute and derived from the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
WEB = ROOT / "web"
for _p in (DATA, CACHE, MODELS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset
HF_DATASET = "westbrook/English_Accent_DataSet"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
N_TRAIN_SHARDS = 28

# The dataset's ClassLabel order. Index == integer label stored in parquet.
ACCENTS = [
    "Dutch", "German", "Czech", "Polish", "French", "Hungarian", "Finnish",
    "Romanian", "Slovak", "Spanish", "Italian", "Estonian", "Lithuanian",
    "Croatian", "Slovene", "English", "Scottish", "Irish", "NorthernIrish",
    "Indian", "Vietnamese", "Canadian", "American",
]

# Human-facing grouping used by the coaching layer.
NATIVE_VARIETIES = {"English", "Scottish", "Irish", "NorthernIrish", "American", "Canadian"}

# ---------------------------------------------------------------- audio
SR = 16000
MIN_DUR = 2.0     # clips shorter than this carry too little accent evidence
MAX_DUR = 12.0    # cap to bound encoder cost; longer clips are centre-cropped

# ---------------------------------------------------------------- models
SSL_ENCODER = "microsoft/wavlm-base-plus"
# Mid layers of an SSL encoder carry the most accent/phonetic information;
# the top layers drift toward the pretraining objective.
SSL_LAYERS = [4, 6, 8, 10, 12]
PHONEME_MODEL = "mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39"
ASR_MODEL = "base.en"          # faster-whisper (CTranslate2), int8 on CPU

HEAD_DIR = MODELS / "accent_head"

# Encoder-side cap: accent evidence saturates well before this, and it bounds
# the dominant cost of the whole pipeline.
ENCODE_MAX_DUR = 6.0
SHARD_DIR = DATA / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
