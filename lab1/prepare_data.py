import math
import os
import urllib.request
import zipfile
import cv2
import glob

DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
ZIP_PATH = "DIV2K_train_HR.zip"
DATA_PATH = "DIV2K_train_256"

# Preprocessing: scaling i center crop do 256x256 (takes a bit of time, like 4 mins so do it once :D)
def preprocess_div2k(input_dir, output_dir, target_size=256):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, "*.png"))

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]

        scale = target_size / min(h, w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        top = (new_h - target_size) // 2
        left = (new_w - target_size) // 2
        img_cropped = img_resized[top:top+target_size, left:left+target_size]

        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_dir, filename), img_cropped)

if __name__ == "__main__":
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("data/")

    preprocess_div2k(
        input_dir="data/DIV2K_train_HR",
        output_dir=os.path.join("data", DATA_PATH)
    )