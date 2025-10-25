import os
from PIL import Image, ImageFilter, ImageChops
import numpy as np
import random
from tqdm import tqdm
import shutil
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

def maybe_blur(img: Image.Image) -> Image.Image:
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

def maybe_noise(img: Image.Image) -> Image.Image:
    if random.random() < 0.3:
        arr = np.array(img)
        arr = np.clip(arr + np.roll(arr, 1, axis=0) + np.roll(arr, 1, axis=1), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img

def augment_image(img: Image.Image) -> Image.Image:
    img = rotate_image(img)
    img = random_shift(img)
    img = random_scale(img)
    img = center_image(img)
    img = maybe_blur(img)
    img = maybe_noise(img)
    return img

def remove_corrupted_images(folder: str):
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception:
                print(f"Supprimée image corrompue : {file_path}")
                os.remove(file_path)

def convert_images_to_grayscale(folder: str):
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path).convert("L")
            img.save(file_path)

def remove_duplicate_images(folder: str):
    hashes = set()
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in hashes:
                print(f"Supprimé doublon : {file_path}")
                os.remove(file_path)
            else:
                hashes.add(file_hash)

def preprocess_and_augment(input_dir: str, output_dir: str, n_augment: int = 5):
    print("Étape 1 : Suppression des images corrompues...")
    remove_corrupted_images(input_dir)
    
    print("Étape 2 : Conversion des images en niveaux de gris...")
    convert_images_to_grayscale(input_dir)
    
    print("Étape 3 : Suppression des doublons...")
    remove_duplicate_images(input_dir)

    print("Étape 4 : Augmentation du dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(input_dir):
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(input_class_dir), desc=f"Classe {class_name}"):
            img_path = os.path.join(input_class_dir, img_name)
            img = Image.open(img_path).convert("L")

            base_name, ext = os.path.splitext(img_name)
            img.save(os.path.join(output_class_dir, f"{base_name}_orig{ext}"))
            for i in range(n_augment):
                aug_img = augment_image(img)
                aug_img.save(os.path.join(output_class_dir, f"{base_name}_aug{i}{ext}"))

    print(f"Pipeline terminé. Dataset final créé dans {output_dir}")


if __name__ == "__main__":
    preprocess_and_augment(
        input_dir="data/train",
        output_dir="data/train_augmented",
        n_augment=5
    )
