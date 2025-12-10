from network import HybirdSegmentationAlgorithm
import torch
from segement import segment_class_on_image, save_segmented_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from set_data_set import parse_image
from utils import profile_block
import os


def test_model_inference(model, image):
    image = image.half()

    outputs = model(image)

    print("test.py - image:")
    print(image)
    print("test.py - outputs[0]:")
    print(outputs[0])
    print("test.py - outputs[1]:")
    print(outputs[1])

    segmented_image, binary_mask = profile_block(
        "segment_class_on_image", segment_class_on_image, outputs, image, class_id=0
    )
    return segmented_image, binary_mask


def test_model(img_path):

    image = torch.from_numpy(parse_image(img_path, size=640, channels=3).numpy()).to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    segmented_image, binary_mask = profile_block(
        "test_model_inference", test_model_inference, model, image
    )

    if binary_mask.dim() == 4:
        if binary_mask.size(1) == 1:
            binary_mask = binary_mask.squeeze(1)  # (B, H, W)
        else:
            # لو multi-channel ناخد أول قناة (أو ممكن نعمل mean)
            binary_mask = binary_mask[:, 0, :, :]  # (B, H, W)

    print("test.py - binary_mask:")
    print(binary_mask.to(device))

    if segmented_image is not None:
        save_segmented_image(segmented_image, "segmented_image.png")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        with torch.cuda.amp.autocast():

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
            folder = "P3M-10k/train/blurred_image"
            sorted_images = sorted(os.listdir(folder))
            profile_block(
                "test model", test_model, os.path.join(folder, sorted_images[0])
            )
