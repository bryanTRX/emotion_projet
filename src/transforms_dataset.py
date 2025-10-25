import os
from PIL import Image, ImageChops
import random
from tqdm import tqdm
import hashlib

def rotate_image(img: Image.Image) -> Image.Image:
    angle = random.uniform(-15, 15)
    return img.rotate(angle, fillcolor=0)

def random_shift(img: Image.Image) -> Image.Image:
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    return ImageChops.offset(img, shift_x, shift_y)

def random_scale(img: Image.Image) -> Image.Image:
    scale = random.uniform(0.9, 1.1)
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

def center_image(img: Image.Image) -> Image.Image:
    img_final = Image.new("L", (48, 48), 0)
    paste_x = max((48 - img.size[0]) // 2, 0)
    paste_y = max((48 - img.size[1]) // 2, 0)
    img_final.paste(img, (paste_x, paste_y))
    return img_final

def augment_image(img: Image.Image) -> Image.Image:
    img = rotate_image(img)
    img = random_shift(img)
    img = random_scale(img)
    img = center_image(img)
    return img

def remove_corrupted_images(folder: str):
    print("\n🧹 Suppression des images corrompues...")
    for root, _, files in tqdm(os.walk(folder), desc="Vérification des fichiers"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception:
                os.remove(file_path)

def convert_images_to_grayscale(folder: str):
    print("\n🎨 Conversion des images en niveaux de gris...")
    for root, _, files in tqdm(os.walk(folder), desc="Conversion en gris"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path).convert("L")
                img.save(file_path)
            except Exception:
                continue

def remove_duplicate_images(folder: str):
    print("\n🧩 Suppression des doublons...")
    hashes = set()
    for root, _, files in tqdm(os.walk(folder), desc="Recherche des doublons"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in hashes:
                    os.remove(file_path)
                else:
                    hashes.add(file_hash)
            except Exception:
                continue

def preprocess_and_augment(input_dir: str, output_dir: str, n_augment: int = 5):
    print("🚀 Démarrage du pipeline de préparation des données")

    remove_corrupted_images(input_dir)
    convert_images_to_grayscale(input_dir)
    remove_duplicate_images(input_dir)

    print("\n📈 Étape finale : Augmentation du dataset...")
    os.makedirs(output_dir, exist_ok=True)

    total_classes = len(os.listdir(input_dir))
    for idx, class_name in enumerate(os.listdir(input_dir), start=1):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        print(f"\n[{idx}/{total_classes}] Classe : {class_name}")
        images = os.listdir(input_class_dir)
        for img_name in tqdm(images, desc=f"Augmentation {class_name}"):
            img_path = os.path.join(input_class_dir, img_name)
            try:
                img = Image.open(img_path).convert("L")
                base_name, ext = os.path.splitext(img_name)

                img.save(os.path.join(output_class_dir, f"{base_name}_orig{ext}"))
                for i in range(n_augment):
                    aug_img = augment_image(img)
                    aug_img.save(os.path.join(output_class_dir, f"{base_name}_aug{i}{ext}"))
            except Exception:
                continue

    print(f"\n✅ Pipeline terminé avec succès ! Dataset final créé dans : {output_dir}")

if __name__ == "__main__":
    preprocess_and_augment(
        input_dir="../data/train",
        output_dir="../data/train_augmented",
        n_augment=5
    )
