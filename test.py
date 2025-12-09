from network import HybirdSegmentationAlgorithm
import torch
from segement import segment_class_on_image, save_segmented_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from utils import profile_block



def test_model_inference(model, image: torch.Tensor):

    outputs = model(image)

    segmented_image, binary_mask = profile_block("segment_class_on_image", segment_class_on_image, outputs, image, class_id=0)
    return segmented_image, binary_mask


def test_model(image: torch.Tensor):
    image = image.unsqueeze(0).to(device)  # (1, 3, 640, 640)

    segmented_image, binary_mask = profile_block("test_model_inference", test_model_inference, model, image)

    if segmented_image is not None:
        save_segmented_image(segmented_image, "segmented_image.png")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HybirdSegmentationAlgorithm(num_classes=1).to(device)
            model = model.eval()

            model.load_state_dict(torch.load("hybrid_seg_single_overfit.pt", map_location="cuda"))

            for module in model.modules():
                if hasattr(module, "fuse"):
                    module.fuse()

            img = Image.open("1755856419306.png").convert("RGB")
            img = TF.resize(
                img,
                (640, 640),
                interpolation=InterpolationMode.BILINEAR,
            )
            img_tensor = TF.to_tensor(img)  # (3, H, W) في الرينج [0,1]

            profile_block("test model", test_model, img_tensor)
            profile_block("test model", test_model, img_tensor)
