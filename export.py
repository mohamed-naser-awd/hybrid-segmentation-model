import os
import torch
from torchvision.utils import save_image
from PIL import Image
from dataset import PetSegmentationDataset  # استورد الكلاس اللي عندك
import torchvision.transforms.functional as TF

# فولدر الإخراج
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# تحميل الداتاسيت (هتنزل أوتوماتيك أول مرة)
dataset = PetSegmentationDataset(root="./data", image_size=640)
print(f"Dataset length: {len(dataset)}")

# ناخد أول صورة وماسك
image, mask = dataset[0]  # image: (3,H,W), mask: (H,W)
print("Loaded sample shapes:", image.shape, mask.shape)

# --- حفظ الصورة ---
save_image(image, os.path.join(EXPORT_DIR, "image.png"))

# --- حفظ الماسك ---
# الماسك قيمها 0 أو 1، نحولها لصورة رمادي
mask_img = TF.to_pil_image(mask)
mask_img.save(os.path.join(EXPORT_DIR, "mask.png"))

print("✅ Saved to 'export/image.png' and 'export/mask.png'")
