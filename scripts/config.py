from pathlib import Path
import os


# ============================================================
# TinyRAG Project Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Main folders
SCRIPTS_DIR = PROJECT_DIR / "scripts"


DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"
INDEXES_DIR = PROJECT_DIR / "indexes"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

# Hugging Face cache
HF_CACHE_DIR = PROJECT_DIR / "hf_cache"

# CPU-only setting
DEVICE = "cpu"

# Reproducibility
SEED = 42


# Create directories if they do not exist
DIRECTORIES = [
    DATA_DIR,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    INDEXES_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    HF_CACHE_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)


# Hugging Face cache locations
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(
    HF_CACHE_DIR / "datasets"
)
os.environ["HF_HUB_CACHE"] = str(
    HF_CACHE_DIR / "hub"
)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Display configuration
print("=" * 60)
print("TinyRAG Configuration")
print("=" * 60)

print("Project directory:")
print(PROJECT_DIR)

print("\nDevice:")
print(DEVICE)

print("\nRandom seed:")
print(SEED)

print("\nConfiguration loaded successfully.")
print("=" * 60)