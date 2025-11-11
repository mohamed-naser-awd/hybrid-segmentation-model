from network import HybirdSegmentationAlgorithm
from PIL import Image
import torch
import torchvision.transforms as T
from segement import segment_class_on_image, save_segmented_image


if __name__ == "__main__":
    model = HybirdSegmentationAlgorithm(num_classes=1)
    model.eval()
    image = Image.open("image.jpg")
    image = image.resize((640, 640))
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    tensor: torch.Tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output: torch.Tensor = model(tensor)

segmented_image, binary_mask = segment_class_on_image(
    output,
    image,
    class_id=1,  # مثلاً 1 = person حسب ترتيب الكلاسات عندك
    score_thresh=0.7,
    mask_thresh=0.5,
)

if segmented_image is not None:
    save_segmented_image(segmented_image, "person_segment.png")
