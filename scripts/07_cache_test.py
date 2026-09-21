from config import HF_CACHE_DIR

import os

print("HF_HOME:")
print(os.environ.get("HF_HOME"))

print("\nHF_HUB_CACHE:")
print(os.environ.get("HF_HUB_CACHE"))

print("\nHF_DATASETS_CACHE:")
print(os.environ.get("HF_DATASETS_CACHE"))

print("\nProject HF cache:")
print(HF_CACHE_DIR)