import torch

def check_cuda() -> None:
    print("🔍 Vérification de la disponibilité CUDA...\n")

    # Vérification de base
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ CUDA disponible")
        print(f"🧠 GPU détecté : {device_name}")
        print(f"🔢 Nombre de GPU : {device_count}")
        print(f"💾 Mémoire totale : {total_mem:.2f} Go")
        print(f"📦 Version PyTorch CUDA : {torch.version.cuda}")
        print(f"⚙️  Version cudNN : {torch.backends.cudnn.version()}")
    elif mps_available:
        print("⚙️  CUDA indisponible, mais MPS (Apple Silicon) détecté.")
        print("✅ Entraînement possible sur GPU Apple.")
    else:
        print("❌ Aucun GPU détecté.")
        print("🧩 Exécution sur CPU uniquement.")

    print("\n🔧 Device utilisé par défaut :", get_best_device())

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    check_cuda()
