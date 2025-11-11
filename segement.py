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
    outputs: dict,
    image: Tensor,  # (B, 3, H, W)
    class_id: int,  # الكلاس اللي عايزه (من 0 لـ num_classes-1)
    score_thresh: float = 0.7,
    mask_thresh: float = 0.5,
):
    """
    بيرجع:
      - segmented_image: صورة جديدة فيها الكلاس ده بس
      - binary_mask: الماسك الباينري (H, W) كـ Tensor
    """
    # هنشتغل على أول صورة بس عشان البساطة
    pred_logits = outputs["pred_logits"][0]  # (Q, num_classes+1)
    pred_masks = outputs["pred_masks"][0]  # (Q, H, W)

    # 1) نجيب probabilities للكلاسات (ونشيل background آخر channel)
    class_probs = pred_logits.softmax(-1)[..., :-1]  # (Q, num_classes)

    # 2) لكل query نعرف أفضل class و السكور بتاعه
    scores, labels = class_probs.max(-1)  # (Q,)

    # 3) نختار الـ queries اللي بتتنبأ بالكلاس اللي عايزينه وبسكور كويس
    keep = (labels == class_id) & (scores > score_thresh)

    if keep.sum() == 0:
        print("No queries for this class with enough confidence.")
        return None, None

    # 4) نجيب masks للـ queries اللي اخترناها
    masks = pred_masks[keep]  # (K, H, W) , K = عدد الqueries اللي اخترناها
    masks = torch.sigmoid(masks)  # نخليها [0,1] بدل logits

    # 5) نعمل combine للـ masks دي (ممكن max أو sum، max أبسط)
    combined_mask = masks.max(0).values  # (H, W)

    # switch logic to only object exist and background is black
    combined_mask = 1 - combined_mask

    # 6) نعمل threshold ونطلع ماسك باينري
    binary_mask = (combined_mask > mask_thresh).float()  # (H, W)

    # 7) نطبق الماسك على الصورة
    # لو الصورة normalized لازم ترجعها 0–1 أو 0–255 قبل الـ to_pil_image
    # هنا بافترض إنها في الرينج [0,1]
    segmented_image = image * binary_mask.unsqueeze(0)  # (3, H, W)

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
