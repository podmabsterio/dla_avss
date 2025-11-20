import torch
from torch import nn

from src.model.base_model import BaseModel
from src.model.rtfs_net_modules.caf_block import CAFBlock
from src.model.rtfs_net_modules.decoder import Decoder
from src.model.rtfs_net_modules.encoders import AudioEncoder
from src.model.rtfs_net_modules.rtfs_block import RTFSBlock
from src.model.rtfs_net_modules.s3_block import S3Block
from src.model.rtfs_net_modules.vp_block import VPBlock


class RTFSNet(nn.Module):
    def __init__(
        self,
        win=256,
        hop=128,
        num_heads_ap=4,
        num_heads_vp=8,
        num_rnn_layers=4,
        num_rtfs_blocks=4,
        use_sru=False,
        share_rtfs_block_weights=True,
    ):
        super().__init__()

        self.share_rtfs_block_weights = share_rtfs_block_weights
        self.num_rtfs_blocks = num_rtfs_blocks
        self.c_a = 256
        self.encoder = AudioEncoder(win=win, hop_length=hop, out_chan=self.c_a)
        self.audio_bottleneck = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=self.c_a),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.c_a, out_channels=self.c_a, kernel_size=1),
        )

        rtfsblock_params = dict(
            in_channels=self.c_a,
            hidden_channels=64,
            rnn_hidden_channels=32,
            q=2,
            num_rnn_layers=num_rnn_layers,
            rnn_kernel_size=8,
            num_feats=win // 2,
            num_heads=num_heads_ap,
            attn_hidden_channels=4,
            use_sru=use_sru,
        )

        self.rtfs_block = RTFSBlock(**rtfsblock_params)

        self.vp_in_chan = 512
        self.vp = VPBlock(
            in_channels=self.vp_in_chan,
            hidden_channels=64,
            q=4,
            num_heads=num_heads_vp,
        )

        self.caf = CAFBlock(chan_audio=self.c_a, chan_video=self.vp_in_chan)

        self.rtfs_block_list = None

        if self.share_rtfs_block_weights:
            self.rtfs_block = RTFSBlock(**rtfsblock_params)
        else:
            self.rtfs_block_list = nn.ModuleList(
                [RTFSBlock(**rtfsblock_params) for _ in range(self.num_rtfs_blocks)]
            )

        self.s3 = S3Block(c_a=self.c_a)
        self.decoder = Decoder(in_chan=self.c_a, hop_length=hop, win=win)

    def _apply_rtfs_blocks(self, x: torch.Tensor, res: torch.Tensor):
        for i in range(self.num_rtfs_blocks):
            if self.share_rtfs_block_weights:
                x = self.rtfs_block(
                    x + res
                )  # TODO differs from original impl when i == 0
            else:
                x = self.rtfs_block_list[i](x + res)
        return x

    def forward(self, x: torch.Tensor, video_emb: torch.Tensor):
        """
        Args:
            x (torch.Tensor): audio batch
            video_emb (torch.Tensor): video embeddings batch
        """
        x = self.encoder(x)
        x_mix_emb = x

        x = self.audio_bottleneck(x)

        x_res = x

        x = self.rtfs_block(x)
        y = self.vp(video_emb)

        x = self.caf(x, y)
        x = self._apply_rtfs_blocks(x, x_res)
        x = self.s3(x_mix_emb, x)

        return self.decoder(x)


class RTFSNet2SpeakersSeparation(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rtfs_net = RTFSNet(*args, **kwargs)

    def forward(self, mix: torch.Tensor, video_emb: torch.Tensor, **kwargs):
        """
        Args:
            mix (torch.Tensor): audio batch
            video_emb (torch.Tensor): video embeddings batch
        """
        preds = self.rtfs_net(mix, video_emb)

        return {"preds": preds}


class RTFSNetVideoEncoding2SpeakersSeparation(
    BaseModel
):  # this version of the model applies video encoder
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rtfs_net = RTFSNet2SpeakersSeparation(*args, **kwargs)

    def forward(self, mix: torch.Tensor, video_emb: torch.Tensor, **kwargs):
        """
        Args:
            mix (torch.Tensor): audio batch
            video_emb (torch.Tensor): video embeddings batch
        """
        # s1_video_emb = video_enc(s1_mouth)
        # s2_video_emb = video_enc(s2_mouth)
        # s1 = self.rtfs_net(mix, s1_video_emb)
        # s2 = self.rtfs_net(mix, s2_video_emb)

        # preds = torch.cat([s1, s2], dim=1)
        # return {'preds': preds} TODO

        raise NotImplementedError()
