# 1. أنشئ نفس الكلاس بالظبط (HybirdSegmentationAlgorithm)
from network import HybirdSegmentationAlgorithm
import torch
from train import load_single_sample
from segement import segment_class_on_image, save_segmented_image
from datetime import datetime

# 2. أنشئ نسخة فاضية بنفس config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybirdSegmentationAlgorithm(num_classes=1, d_model=384).to(device)


# 3. حمّل الـ state_dict
model.load_state_dict(torch.load("hybrid_seg_single_overfit.pt", map_location="cuda"))

# 4. خليه في وضع eval
model.eval()

_image, mask = load_single_sample(
    image_path="export/image.png", mask_path="export/mask.png"
)

def test_model(image: torch.Tensor):
    image = image.unsqueeze(0).to(device)  # (1, 3, 640, 640)

    start_time = datetime.now()

    outputs = model(image)
    model_end_time = datetime.now()
    print(f"Model Time taken: {(model_end_time - start_time).total_seconds()} seconds")
    print(outputs["pred_logits"].shape, "is the shape of the pred_logits")

    segmented_image, binary_mask = segment_class_on_image(outputs, image, class_id=0)

    end_time = datetime.now()
    print(f"Time taken: {(end_time - start_time).total_seconds()} seconds")

    if segmented_image is not None:
        save_segmented_image(segmented_image, "segmented_image.png")


if __name__ == "__main__":
    test_model(_image)
    test_model(_image)
    test_model(_image)
    test_model(_image)
    test_model(_image)
