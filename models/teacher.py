from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F


class BiRefNetTeacher:
    """
    BiRefNet - Bilateral Reference Network
    SOTA for high-resolution dichotomous image segmentation
    Excellent for hair, fingers, and fine edge details
    """

    def __init__(self, device="cuda"):
        from transformers import AutoModelForImageSegmentation, AutoProcessor

        self.device = device
        # BiRefNet-general is best for humans
        model_id = "ZhengPeng7/BiRefNet"

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

        # BiRefNet expects normalized images
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def predict_soft_mask(self, img: Image.Image, size=640) -> torch.Tensor:
        """Returns soft probability mask [1, H, W] in float16"""
        # Resize image
        img_resized = img.resize((size, size), Image.BILINEAR)

        # To tensor and normalize
        img_tensor = (
            TF.to_tensor(img_resized).unsqueeze(0).to(self.device)
        )  # [1, 3, H, W]
        img_tensor = (img_tensor - self.mean) / self.std

        # Inference
        outputs = self.model(img_tensor)

        # BiRefNet returns list of predictions at different scales
        # Take the final refined output
        if isinstance(outputs, (list, tuple)):
            pred = outputs[-1]  # Final output
        else:
            pred = outputs

        # Sigmoid to get probabilities
        soft_mask = torch.sigmoid(pred)

        # Ensure correct size
        if soft_mask.shape[-2:] != (size, size):
            soft_mask = F.interpolate(
                soft_mask, size=(size, size), mode="bilinear", align_corners=False
            )

        return soft_mask[0].cpu().half()  # [1, H, W]


class InSPyReNetTeacher:
    """
    InSPyReNet - Inverse Saliency Pyramid Reconstruction Network
    Very accurate, good balance of speed and quality
    """

    def __init__(self, device="cuda", mode="base"):
        from transparent_background import Remover

        # mode: 'fast', 'base', 'base-nightly' (base-nightly is most accurate)
        self.remover = Remover(mode=mode, device=device)
        self.device = device

    @torch.no_grad()
    def predict_soft_mask(self, img: Image.Image, size=640) -> torch.Tensor:
        """Returns soft probability mask [1, H, W] in float16"""
        # Get soft mask (returns PIL Image in 'map' mode)
        mask = self.remover.process(img, type="map")

        # Resize to target size
        mask = mask.resize((size, size), Image.BILINEAR)

        # To tensor
        mask_tensor = TF.to_tensor(mask)  # [1, H, W] or [3, H, W]

        # Ensure single channel
        if mask_tensor.shape[0] == 3:
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)

        return mask_tensor.half()


class SegformerHumanTeacher:
    """
    SegFormer fine-tuned on human parsing datasets
    Good for multi-part human segmentation
    """

    def __init__(self, device="cuda"):
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.device = device
        # Human parsing model
        model_id = "matei-dorian/segformer-b5-finetuned-human-parsing"

        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_soft_mask(self, img: Image.Image, size=640) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits

        # Upsample
        logits = F.interpolate(
            logits, size=(size, size), mode="bilinear", align_corners=False
        )

        # Sum all human part classes (exclude background class 0)
        probs = torch.softmax(logits, dim=1)
        human_prob = probs[0, 1:].sum(dim=0, keepdim=True)  # Sum all non-background
        human_prob = human_prob.clamp(0, 1)

        return human_prob.cpu().half()
