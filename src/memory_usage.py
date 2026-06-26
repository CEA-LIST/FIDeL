import torch
from tqdm import tqdm
import psutil

# Essaye d'importer pynvml pour le suivi de la mémoire GPU
try:
    import pynvml

    pynvml.nvmlInit()
    gpu_available = True
except Exception as e:
    print("pynvml non disponible ou GPU non détecté:", e)
    gpu_available = False


def get_cpu_memory_usage():
    """Retourne l'occupation de la mémoire CPU en pourcentage."""
    mem = psutil.virtual_memory()
    return mem.percent


def get_gpu_memory_usage():
    """Retourne l'occupation de la mémoire GPU en pourcentage (pour le GPU 0)."""
    if gpu_available:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / mem_info.total * 100
    else:
        return None
