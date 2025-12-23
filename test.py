import os
from network import SemanticSegmentationModel
from segement import save_segmented_image
from set_data_set import parse_image
from utils import profile_block
import torch


def test_model_inference(model, image):
    """
    يعمل inference على صورة واحدة:
    - يحوّلها لنفس الـ device
    - يشغّل الموديل
    - يطبّق segment_all_objects
    """
    logits = model(image)
    return torch.sigmoid(logits)


def test_model(img_path: str, save_path: str = "exported_images"):
    """
    - يقرأ الصورة من المسار
    - يعمل لها parsing بنفس دالة الـ dataset (parse_image)
    - يشغّل test_model_inference
    - يحفظ الناتج كـ segmented_image.png
    """
    # parse_image بيرجع Tensor [3, H, W] بقيم [0, 1]
    image = parse_image(img_path, size=640, channels=3)  # [3, H, W]
    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    probs = profile_block(
        "test_model_inference",
        test_model_inference,
        model,
        image,
    )

    threshold = 0.5
    binary_mask = (probs > threshold).float()
    segmented_image = image * binary_mask

    save_path = os.path.join(save_path, f"{img_path.split('/')[-1].split('.')[0]}.png")
    save_segmented_image(segmented_image, save_path)
    print(f"Segmented image saved to {save_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # حمّل الموديل
    model = SemanticSegmentationModel(
        num_classes=1,
        query_count=1,  # مهم: semantic mask واحدة
    ).to(device)
    state_dict = torch.load("model.pt", map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    for image in os.listdir("images"):
        img_path = os.path.join("images", image)
        with torch.no_grad():
            # نستخدم autocast للـ FP16 (mixed precision) في inference
            # with torch.cuda.amp.autocast(dtype=torch.float16):
            profile_block("test_model", test_model, img_path)
