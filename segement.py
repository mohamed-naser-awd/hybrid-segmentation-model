"""
What is going on here?
This file contains the code for segmenting a class on an image using a model.
We have two functions:
1. segment_class_on_image: This function segments a class on an image using a model.
2. save_segmented_image: This function saves the segmented image to a file.

how segment_class_on_image works?
we take model outputs and original image
model outputs are pred_logits and pred_masks

a class can have multiple masks, so we need to combine the masks for the same class
logits is Q, num_classes+1 tensor, so we need to get the class with the highest probability

each pred_logits i has its own mask of index(i)
we first need to get logits for the class we want to segment
we know which masks we have then
we combine the masks for the same class so we get max probability from all masks 

not every pixel in mask belong to the class as its probability is not 1
so we need to threshold the mask to get the binary mask
we use mask_thresh to get the binary mask
we use score_thresh to get the keep mask using the logits

we create binary_mask by comparing the combined_mask with mask_thresh giving zeroes and ones matrix

we just multiply the image with the binary_mask to get the segmented image

be careful, training dataset the object is segmented with the mask, so generated image is the background as it is and the object is black

to switch so only object exist and background is black, we need to subtract the binary_mask from 1


"""


import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


def segment_class_on_image(
    outputs,
    image: Tensor,  # (B, 3, H, W) أو (3, H, W)
    class_id: int,
    score_thresh: float = 0.7,
    mask_thresh: float = 0.5,
):
    pred_logits, pred_masks = outputs

    # --- اشتغل على أول صورة بس في الباتش ---
    if pred_logits.dim() == 3:
        # (B, Q, C1) -> (Q, C1)
        pred_logits = pred_logits[0]
    if pred_masks.dim() == 4:
        # (B, Q, H, W) -> (Q, H, W)
        pred_masks = pred_masks[0]

    # image برضه نخليها (3, H, W)
    if image.dim() == 4:
        image = image[0]

    # 1) بروبس لكل كلاس (ونشيل background آخر channel)
    class_probs = pred_logits.softmax(-1)[..., :-1]  # (Q, num_classes)

    # 2) أفضل كلاس وسكور لكل query
    scores, labels = class_probs.max(-1)  # (Q,)

    # دي مهمة عشان تشوف ليه مفيش queries
    print("scores:", scores.detach().cpu())
    print("labels:", labels.detach().cpu())

    # 3) queries اللي بتتكلم عن الكلاس المطلوب وبسكور كويس
    keep = (labels == class_id) & (scores > score_thresh)

    if keep.sum() == 0:
        print("No queries for this class with enough confidence.")
        return None, None

    # 4) masks للـ queries المختارة
    masks = pred_masks[keep]          # (K, H, W)
    masks = torch.sigmoid(masks)      # logits -> [0,1]

    # 5) combine (max أو sum، max كويس هنا)
    combined_mask = masks.max(0).values  # (H, W)

    # 6) threshold للماسكات
    binary_mask = (combined_mask > mask_thresh).float()  # (H, W)

    # 7) apply على الصورة (3, H, W)
    segmented_image = image * binary_mask.unsqueeze(0)

    return segmented_image, binary_mask


def save_segmented_image(segmented_image: torch.Tensor, path: str):
    # لو داخل (B, C, H, W) أو (B, H, W) → شيل الـ batch
    if segmented_image.dim() == 4:
        segmented_image = segmented_image.squeeze(0)

    # لو داخل (H, W) → خليه (1, H, W) عشان to_pil_image يفهمه كـ grayscale
    if segmented_image.dim() == 2:
        segmented_image = segmented_image.unsqueeze(0)

    # تأكد إن القيم بين 0 و 1
    segmented_image = segmented_image.clamp(0, 1)

    img_pil = to_pil_image(segmented_image.cpu())
    img_pil.save(path)
