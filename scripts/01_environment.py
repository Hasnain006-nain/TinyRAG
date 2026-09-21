import sys
import platform
import psutil
import torch


def main():
    print("=" * 60)
    print("TinyRAG CPU Environment Check")
    print("=" * 60)

    print("\nPython")
    print("Version:", sys.version)
    print("Executable:", sys.executable)

    print("\nSystem")
    print("OS:", platform.platform())
    print("CPU:", platform.processor())

    print("\nCPU")
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Logical processors:", psutil.cpu_count(logical=True))

    print("\nMemory")
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    available_ram = psutil.virtual_memory().available / (1024 ** 3)

    print(f"Total RAM: {total_ram:.2f} GB")
    print(f"Available RAM: {available_ram:.2f} GB")

    print("\nPyTorch")
    print("Version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    print("\nSelected device: CPU")

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()