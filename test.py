import os
from network import HybirdSegmentationAlgorithm
from segement import save_segmented_image
from set_data_set import parse_image
from utils import profile_block
import torch
import torch.nn.functional as F

FILL_HOLES = True

def _morph_dilate(x: torch.Tensor, k: int, iters: int = 1) -> torch.Tensor:
    # x: [1,1,H,W] float(0/1)
    pad = k // 2
    w = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype)
    for _ in range(iters):
        y = F.conv2d(x, w, padding=pad)
        x = (y > 0).to(x.dtype)
    return x


def _morph_erode(x: torch.Tensor, k: int, iters: int = 1) -> torch.Tensor:
    pad = k // 2
    w = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype)
    full = float(k * k)
    for _ in range(iters):
        y = F.conv2d(x, w, padding=pad)
        x = (y >= full - 1e-6).to(x.dtype)
    return x


def _closing(x: torch.Tensor, k: int, iters: int = 1) -> torch.Tensor:
    return _morph_erode(_morph_dilate(x, k, iters), k, iters)


def _fill_holes_cpu(mask_2d_u8, max_hole_area: int):
    """
    mask_2d_u8: numpy uint8 {0,255}
    يملأ الثقوب الداخلية (التي لا تتصل بحافة الصورة) حتى مساحة معينة.
    """
    import numpy as np
    from collections import deque

    H, W = mask_2d_u8.shape
    bg = mask_2d_u8 == 0
    visited = np.zeros((H, W), dtype=np.uint8)

    # 1) علّم الخلفية المتصلة بحواف الصورة (مش holes)
    q = deque()
    for x in range(W):
        if bg[0, x]:
            q.append((0, x))
            visited[0, x] = 1
        if bg[H - 1, x]:
            q.append((H - 1, x))
            visited[H - 1, x] = 1
    for y in range(H):
        if bg[y, 0]:
            q.append((y, 0))
            visited[y, 0] = 1
        if bg[y, W - 1]:
            q.append((y, W - 1))
            visited[y, W - 1] = 1

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while q:
        y, x = q.popleft()
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = 1
                q.append((ny, nx))

    # 2) أي خلفية غير متعلمة = hole. املأها لو مساحتها <= max_hole_area
    holes = bg & (visited == 0)
    # نفك holes إلى components ونقيس المساحة
    visited2 = np.zeros((H, W), dtype=np.uint8)

    for y0 in range(H):
        for x0 in range(W):
            if holes[y0, x0] and not visited2[y0, x0]:
                comp = []
                q = deque([(y0, x0)])
                visited2[y0, x0] = 1
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and holes[ny, nx]
                            and not visited2[ny, nx]
                        ):
                            visited2[ny, nx] = 1
                            q.append((ny, nx))

                if len(comp) <= max_hole_area:
                    for yy, xx in comp:
                        mask_2d_u8[yy, xx] = 255  # fill

    return mask_2d_u8


def postprocess_mask(binary_mask: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    binary_mask: [1,H,W] float/bool (0/1)
    returns: [1,H,W] float(0/1)
    """
    # defaults
    close_k = int(cfg.get("close_kernel", 0))  # 0 = off
    close_iters = int(cfg.get("close_iters", 1))
    fill_holes = bool(cfg.get("fill_holes", True))
    max_hole_area = int(cfg.get("max_hole_area", 1500))  # adjust حسب دقتك
    bridge_k = int(cfg.get("bridge_kernel", 0))  # optional extra bridging
    bridge_iters = int(cfg.get("bridge_iters", 1))

    x = binary_mask
    if x.dim() == 3:
        x = x.unsqueeze(1)  # [1,1,H,W]
    x = (x > 0.5).to(torch.float32)

    # (A) Closing خفيف لتقليل عضّ الذراع/حواف مكسورة
    if close_k and close_k >= 3:
        x = _closing(x, close_k, close_iters)

    # (B) Gap bridging إضافي (اختياري) لو لسه في تقطيع بسيط
    if bridge_k and bridge_k >= 3:
        x = _morph_dilate(x, bridge_k, bridge_iters)
        x = _morph_erode(x, bridge_k, bridge_iters)

    # (C) Fill holes المشروط بالمساحة وبأنه hole داخلي
    if fill_holes:
        # CPU خطوة واحدة فقط (آمنة ودقيقة)
        m = (x[0, 0].detach().to("cpu") * 255).to(torch.uint8).numpy()
        m = _fill_holes_cpu(m, max_hole_area=max_hole_area)
        x = torch.from_numpy((m > 0).astype("float32")).to(binary_mask.device)[
            None, None, :, :
        ]

    return x[:, 0]  # [1,H,W]


def attach_nan_debug_hooks(model):
    """
    يضيف forward hooks على كل الموديولز عشان يطبع أي NaN/Inf في الـ outputs.
    """

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
                    print(
                        f"[NaN/Inf AFTER] {name} (tensor {i}) "
                        f"min={t.min().item()} max={t.max().item()} "
                        f"NaN={has_nan} Inf={has_inf}"
                    )

        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(make_hook(name))


def segment_all_objects(outputs, image, mask_threshold: float = 0.9):
    """
    يدمج كل الماسكات من كل الكويريز في ماسك واحد، ويطبقه على الصورة.

    outputs:
        query_classes: [B, Q, C] (مش بنستخدمها هنا، بنستعمل كل الكويريز)
        masks       : [B, Q, H, W]
    image:
        [B, 3, H, W] أو [3, H, W]

    يرجّع:
        segmented_image: [1, 3, H, W]
        binary_mask    : [1, H, W]
    """
    query_classes, masks = outputs  # shapes: [B, Q, C], [B, Q, H, W]
    # masks = refine_mask(masks)

    # نتأكد إن للصورة batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, 3, H, W]

    image = image.to(masks.device)

    B, Q, H, W = masks.shape
    assert B == 1, "segment_all_objects حالياً متظبط لـ batch size = 1 في inference."

    # ناخد ماسكات كل الكويريز للصورة الوحيدة
    masks_queries = masks[0]  # [Q, H, W]
    masks_probs = masks_queries.sigmoid()  # [Q, H, W]

    # ندمج كل الماسكات بأقصى قيمة (OR تقريباً)
    merged_mask, _ = torch.max(masks_probs, dim=0, keepdim=True)  # [1, H, W]

    # binary mask
    binary_mask = (merged_mask > mask_threshold).to(image.dtype)  # [1, H, W]

    if FILL_HOLES:
        pp_cfg = {
            "close_kernel": 3,  # 0 لإيقافه
            "close_iters": 1,
            "fill_holes": True,
            "max_hole_area": 1200,  # جرب 800-3000 حسب 640x640
            "bridge_kernel": 0,  # خليه 0 مبدئياً
            "bridge_iters": 1,
        }

        binary_mask = postprocess_mask(binary_mask, pp_cfg).to(image.dtype)  # [1,H,W]

    # نخليها [1, 1, H, W] عشان الـ broadcast مع [1, 3, H, W]
    binary_mask_4d = binary_mask.unsqueeze(1)  # [1, 1, H, W]

    # نضرب الماسك في الصورة: ده هيطلع صورة فيها كل الكائنات في صورة واحدة
    segmented_image = image * binary_mask_4d  # [1, 3, H, W]

    return segmented_image, binary_mask


def test_model_inference(model, image, mask_threshold: float = 0.7):
    """
    يعمل inference على صورة واحدة:
    - يحوّلها لنفس الـ device
    - يشغّل الموديل
    - يطبّق segment_all_objects
    """
    image = image.to(device)

    segmented_image, binary_mask = profile_block(
        "segment_all_objects",
        segment_all_objects,
        model(image),
        image,
        mask_threshold,
    )

    return segmented_image, binary_mask


def test_model(img_path: str, mask_threshold: float = 0.7):
    """
    - يقرأ الصورة من المسار
    - يعمل لها parsing بنفس دالة الـ dataset (parse_image)
    - يشغّل test_model_inference
    - يحفظ الناتج كـ segmented_image.png
    """
    # parse_image بيرجع Tensor [3, H, W] بقيم [0, 1]
    image = parse_image(img_path, size=640, channels=3)  # [3, H, W]

    if image.dim() == 3:
        image = image.unsqueeze(0)

    segmented_image, binary_mask = profile_block(
        "test_model_inference",
        test_model_inference,
        model,
        image,
        mask_threshold,
    )

    if segmented_image is None:
        print("No objects were segmented (mask was empty).")
        return

    # لو حابب تتعامل مع الماسك (مثلاً تطبعه أو تحفظه)
    if binary_mask is not None:
        # binary_mask شكلها [1, H, W]، ممكن نخليها [H, W] لو محتاج
        mask_2d = binary_mask[0]  # [H, W]
        print("Mask stats -> min:", mask_2d.min().item(), "max:", mask_2d.max().item())

    # حفظ الصورة الناتجة (الـ save_segmented_image تتكفّل بالـ CPU/convert لو معمول فيها كده)
    save_path = os.path.join(
        "exported_images",
        f"{img_path.split('/')[-1].split('.')[0]}.png"
    )
    save_segmented_image(segmented_image, save_path)
    print(f"Segmented image saved to {save_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # حمّل الموديل
    model = HybirdSegmentationAlgorithm(num_classes=1).to(device)
    state_dict = torch.load("model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Hooks debug للـ NaN/Inf
    attach_nan_debug_hooks(model)

    for image in os.listdir("images"):
        img_path = os.path.join("images", image)
        with torch.no_grad():
            # نستخدم autocast للـ FP16 (mixed precision) في inference
            with torch.cuda.amp.autocast(dtype=torch.float16):
                profile_block("test_model", test_model, img_path)
