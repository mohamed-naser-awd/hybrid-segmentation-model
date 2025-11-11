from backbone import DarkNet53
from fpn import FPN
from torch import nn, Tensor, einsum, zeros
from torch.nn import functional as F
from datetime import datetime


class HybirdSegmentationAlgorithm(nn.Module):
    def __init__(self, num_classes: int, *args, d_model: int = 384, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet53()
        self.fpn = FPN()
        self.patchify = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=6, batch_first=True),
            num_layers=6,
        )
        self.query_embed = nn.Embedding(num_embeddings=100, embedding_dim=d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.pixel_proj = nn.Conv2d(d_model, d_model, 1)
        self.classifier = nn.Linear(d_model, num_classes + 1)
        self.pos_embed = nn.Parameter(zeros(1, 10000, d_model))

    def forward(self, image: Tensor):
        start_time = datetime.now()
        c3, c4, c5 = self.backbone(image)
        end_time = datetime.now()
        print(f"Backbone Time taken: {(end_time - start_time).total_seconds()} seconds")
        start_time = datetime.now()
        p3, p4, p5 = self.fpn((c3, c4, c5))
        end_time = datetime.now()
        print(f"FPN Time taken: {(end_time - start_time).total_seconds()} seconds")
        p3: Tensor = self.patchify(p3)
        end_time = datetime.now()
        print(f"Patchify Time taken: {(end_time - start_time).total_seconds()} seconds")
        start_time = datetime.now()

        tokens = self.get_tensor_tokens(p3)
        B, N, D = tokens.shape

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        pos = self.pos_embed[:, :N, :]             # (1, N, D)

        start_time = datetime.now()
        decoder_output: Tensor = self.decoder(queries, tokens + pos)
        end_time = datetime.now()
        print(f"Decoder Time taken: {(end_time - start_time).total_seconds()} seconds")

        queries_class = self.classifier(decoder_output)

        queries_proj = self.query_proj(decoder_output)
        pixel_feats = self.pixel_proj(p3)

        masks = einsum("bqd,bdhw->bqhw", queries_proj, pixel_feats)

        masks = F.interpolate(
            masks, size=image.shape[-2:], mode="bilinear", align_corners=False
        )

        return {
            "pred_logits": queries_class,
            "pred_masks": masks,
        }

    def get_tensor_tokens(self, tensor: Tensor):
        tokens = tensor.flatten(2).transpose(1, 2)
        return tokens
