from backbone import DarkNet
from fpn import FPN
from torch import nn, Tensor, einsum, zeros
from torch.nn import functional as F
from utils import profile_block


class HybirdSegmentationAlgorithm(nn.Module):
    def __init__(
        self,
        num_classes: int,
        net_type: str = "18",
        *args,
        d_model: int = 192,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet(net_type=net_type)
        self.fpn = FPN()
        self.patchify = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=3, batch_first=True),
            num_layers=3,
        )

        self.num_queries = 5
        self.query_embed = nn.Embedding(num_embeddings=self.num_queries, embedding_dim=d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.pixel_proj = nn.Conv2d(d_model, d_model, 1)
        self.classifier = nn.Linear(d_model, num_classes + 1)
        self.pos_embed = nn.Parameter(zeros(1, 10000, d_model))
        self.upsample = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)


    def forward(self, image: Tensor):

        c3, c4, c5 = profile_block("backbone", self.backbone, image)
        p3, p4, p5 = profile_block("fpn", self.fpn, (c3, c4, c5))
        p3: Tensor = profile_block("patchify", self.patchify, p3)

        # c3, c4, c5 = self.backbone(image)
        # p3, p4, p5 = self.fpn((c3, c4, c5))
        # p3: Tensor = self.patchify(p3)

        tokens = self.get_tensor_tokens(p3)
        B, N, D = tokens.shape

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        pos = self.pos_embed[:, :N, :]

        decoder_output: Tensor = profile_block(
            "decoder", self.decoder, queries, tokens + pos
        )
        queries_class = self.classifier(decoder_output)

        queries_proj = self.query_proj(decoder_output)
        pixel_feats = self.pixel_proj(p3)

        pixel_feats = self.upsample(pixel_feats)

        masks = einsum("bqd,bdhw->bqhw", queries_proj, pixel_feats)


        return queries_class, masks

    def get_tensor_tokens(self, tensor: Tensor):
        tokens = tensor.flatten(2).transpose(1, 2)
        return tokens
