# 1. أنشئ نفس الكلاس بالظبط (HybirdSegmentationAlgorithm)
from network import HybirdSegmentationAlgorithm
import torch
from train import load_single_sample
from segement import segment_class_on_image, save_segmented_image
from datetime import datetime
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np



def test_model(image: torch.Tensor):
    image = image.unsqueeze(0).to(device)  # (1, 3, 640, 640)

    start_time = datetime.now()
    outputs = model(image)
    model_end_time = datetime.now()

    print(f"Model Time taken: {(model_end_time - start_time).total_seconds()} seconds")

    segmented_image, binary_mask = segment_class_on_image(outputs, image, class_id=0)

    end_time = datetime.now()
    print(f"Time taken: {(end_time - start_time).total_seconds()} seconds")

    if segmented_image is not None:
        save_segmented_image(segmented_image, "segmented_image.png")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HybirdSegmentationAlgorithm(num_classes=1).to(device)
            model = model.eval()

            model.load_state_dict(torch.load("hybrid_seg_p3m10k.pt", map_location="cuda"))

            for module in model.modules():
                if hasattr(module, "fuse"):
                    module.fuse()

            img = Image.open("test.jpg").convert("RGB")
            img = TF.resize(
                img,
                (640, 640),
                interpolation=InterpolationMode.BILINEAR,
            )
            img_tensor = TF.to_tensor(img)  # (3, H, W) في الرينج [0,1]

            test_model(img_tensor)
