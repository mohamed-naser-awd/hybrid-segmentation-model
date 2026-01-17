import os
import torchvision.transforms.functional as TF
from segement import save_segmented_image
from utils import profile_block, pad_to_valid_size
import torch
from models import UltraFastNet as Model, BiRefNetTeacher

import cv2
import logging

logging.getLogger().setLevel(logging.INFO)


def test_model_inference(model, image):
    """
    يعمل inference على صورة واحدة:
    - يحوّلها لنفس الـ device
    - يشغّل الموديل
    - يطبّق segment_all_objects
    """
    logits = model.predict_soft_mask(image)
    # return logits
    return torch.sigmoid(logits)


def test_model(img_path: str, save_path: str = "test_output"):
    """
    - يقرأ الصورة من المسار
    - يشغّل test_model_inference
    - يحفظ الناتج كـ segmented_image.png
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB

    image = pad_to_valid_size(
        image,
        valid_sizes=(
            640,
            1024,
            1280,
            1600,
            1920,
            2048,
            2240,
            2560,
            2880,
            3200,
            3520,
            3840,
            4160,
            4480,
            4800,
            5120,
            5440,
            5760,
            6080,
            6400,
        ),
    )

    image = TF.to_tensor(image).to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    probs = profile_block(
        "test_model_inference",
        test_model_inference,
        model,
        image,
        extra_info=f"Image Shape: {image.shape}",
    )

    threshold = 0.5
    binary_mask = (probs > threshold).float()

    segmented_image = image * binary_mask

    image_name = img_path.split("/")[-1]
    filename = f"{image_name.split('.')[0]}.png"
    input_save_path = os.path.join(save_path, filename)
    save_segmented_image(segmented_image, input_save_path)
    save_segmented_image(
        image, os.path.join(save_path, f"{image_name}_original_padded.jpg")
    )
    print(
        f"Segmented image saved to {input_save_path} and {os.path.join(save_path, f"{image_name}_original_padded.jpg")}"
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # حمّل الموديل
    model = BiRefNetTeacher(device=device)
    if hasattr(model, "eval"):
        model.eval()

    # checkpoint = torch.load("checkpoints_distill/distill_epoch_027.pt")
    # model.load_state_dict(checkpoint["model"])

    for image in os.listdir("images"):
        img_path = os.path.join("images", image)
        with torch.no_grad():
            # نستخدم autocast للـ FP16 (mixed precision) في inference
            # with torch.cuda.amp.autocast(dtype=torch.float16):
            # for _ in range(5):
            test_model(img_path)
