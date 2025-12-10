from network import HybirdSegmentationAlgorithm
import torch
from segement import segment_class_on_image, save_segmented_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from set_data_set import parse_image
from utils import profile_block


def test_model_inference(model, image):

    outputs = model(image)

    segmented_image, binary_mask = profile_block(
        "segment_class_on_image", segment_class_on_image, outputs, image, class_id=0
    )
    return segmented_image, binary_mask


def test_model(img_path):

    image = parse_image(img_path, size=640, channels=3).to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    segmented_image, binary_mask = profile_block(
        "test_model_inference", test_model_inference, model, image
    )

    print(binary_mask)

    if segmented_image is not None:
        save_segmented_image(segmented_image, "segmented_image.png")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HybirdSegmentationAlgorithm(num_classes=1, net_type="18").to(device)
            model = model.eval()

            model.load_state_dict(
                torch.load("hybrid_seg_p3m10k_dark18.pt", map_location="cuda")
            )

            for module in model.modules():
                if hasattr(module, "fuse"):
                    module.fuse()

            # profile_block("test model", test_model, "1755856419306.png")
            profile_block(
                "test model", test_model, "P3M-10k/train/blurred_image/p_0a0c9250.jpg"
            )
