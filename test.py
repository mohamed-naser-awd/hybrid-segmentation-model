from network import HybirdSegmentationAlgorithm
import torch
from segement import segment_class_on_image, save_segmented_image
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from set_data_set import parse_image
from utils import profile_block
import os

def attach_nan_debug_hooks(model):
    def make_hook(name):
        def hook(module, inputs, output):
            tensors = []
            if isinstance(output, torch.Tensor):
                tensors = [output]
            elif isinstance(output, (tuple, list)):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]

            for i, t in enumerate(tensors):
                if t.numel() == 0:
                    continue
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                if has_nan or has_inf:
                    print(f"[NaN/Inf AFTER] {name} (tensor {i}) "
                          f"min={t.min().item()} max={t.max().item()} "
                          f"NaN={has_nan} Inf={has_inf}")
        return hook

    for name, module in model.named_modules():
        # ممكن تستثني حاجات بسيطة لو حابب
        module.register_forward_hook(make_hook(name))


def test_model_inference(model, image):
    image = image.half()

    print("test.py - image:")
    print(image.shape)

    outputs = model(image)

    queries_class, masks = outputs

    print("test.py - queries_class:")
    print(queries_class)
    print("test.py - masks:")
    print(masks)

    binary_mask = (masks.sigmoid() > 0.5).float()
    segmented_image = image * binary_mask

    return segmented_image, binary_mask


def test_model(img_path):

    image = torch.from_numpy(parse_image(img_path, size=640, channels=3).numpy()).to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    segmented_image, binary_mask = profile_block(
        "test_model_inference", test_model_inference, model, image
    )

    return

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybirdSegmentationAlgorithm(num_classes=1, net_type="18").to(device)
    model.load_state_dict(
        torch.load("hybrid_seg_p3m10k_dark18.pt", map_location="cuda")
    )
    model = model.eval()

    attach_nan_debug_hooks(model)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            profile_block("test model", test_model, "test.jpg")
