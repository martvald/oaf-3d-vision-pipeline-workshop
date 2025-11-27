from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn

# ... existing code ...


class SelfAttn(nn.Module):
    """Self Attention Module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim_expansion * dim // self.num_heads
        self.scale: float = self.head_dim**-0.5
        self.use_pe: bool = use_pe
        self.q: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=False)

        self.v: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj: nn.Linear = nn.Linear(dim_expansion * dim, dim, bias=False)
        if self.use_pe:
            self.pe_proj: nn.Linear = nn.Linear(32, self.head_dim)

    def forward(self, x: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        # x: [B, N, C]
        B, N, _ = x.shape
        scale: float = self.scale

        q: Tensor = (
            self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )  # B num_head N head_dim
        k: Tensor = (
            self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        v: Tensor = (
            self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )

        if self.use_pe:
            score: Tensor = torch.einsum(
                "...ic, ...jc -> ...ij", scale * q, k
            )  # B num_head N N
            attn: Tensor = score.reshape(B, self.num_heads, N, N).softmax(dim=-1)
            out: Tensor = torch.einsum("...ij, ...jc -> ...ic", attn, v)
            # add contextual relative positional encoding
            pe_sum: Tensor = torch.einsum("...nij, ijc -> ...nic", attn, pe)
            out = out + self.pe_proj(pe_sum)
        else:
            out: Tensor = cp.checkpoint(  # type: ignore[reportAssignmentType,no-redef]
                F.scaled_dot_product_attention, q, k, v, use_reentrant=False
            )

        out = self.proj(
            out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        )

        return out


class CrossAttn(nn.Module):
    """Symmetric Cross Attention Module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim_expansion * dim // self.num_heads
        self.scale: float = self.head_dim**-0.5
        self.q: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v: nn.Linear = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj: nn.Linear = nn.Linear(dim_expansion * dim, dim, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:

        # x, y: [B, N, C], [B, N, C]
        _, N, _ = x.shape
        B, _, _ = y.shape

        qx: Tensor = (
            self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        ky: Tensor = (
            self.k(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        vy: Tensor = (
            self.v(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        x_out: Tensor = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            qx, ky, vy
        )

        kx: Tensor = (
            self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        qy: Tensor = (
            self.q(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        vx: Tensor = (
            self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        y_out: Tensor = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            qy, kx, vx
        )

        x_out = self.proj(
            x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        )
        y_out = self.proj(
            y_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        )

        return x_out, y_out


class SelfAttnBlock1D(nn.Module):
    """1D Self Attention Block (pre-norm type)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.attn: SelfAttn = SelfAttn(
            dim=self.dim,
            num_heads=self.num_heads,
            dim_expansion=dim_expansion,
            use_pe=use_pe,
        )
        self.norm_pre: nn.LayerNorm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:

        B, H, W, C = z.shape
        z = z.reshape(B * H, W, C)

        z_norm: Tensor = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z

        z = z.reshape(B, H, W, C)
        return z


class CrossAttnBlock1D(nn.Module):
    """1D Cross Attention Block (pre-norm type)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.attn: CrossAttn = CrossAttn(
            self.dim, self.num_heads, dim_expansion=dim_expansion
        )

        self.norm_pre: nn.LayerNorm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:

        z_norm: Tensor = self.norm_pre(z)
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B * H, W, C), y.reshape(B * H, W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z


class SelfAttnBlock2D(nn.Module):
    """2D Self Attention Block (pre-norm type)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.attn: SelfAttn = SelfAttn(
            dim=dim,
            num_heads=num_heads,
            dim_expansion=dim_expansion,
            use_pe=use_pe,
        )
        self.norm_pre: nn.LayerNorm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:

        B, H, W, C = z.shape
        z = z.reshape(B, H * W, C).contiguous()
        z_norm: Tensor = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z
        z = z.reshape(B, H, W, C).contiguous()

        return z


class CrossAttnBlock2D(nn.Module):
    """2D Cross Attention Block (pre-norm type)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.attn: CrossAttn = CrossAttn(
            self.dim, self.num_heads, dim_expansion=dim_expansion
        )
        self.norm_pre: nn.LayerNorm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:

        z_norm: Tensor = self.norm_pre(z)
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B, H * W, C), y.reshape(B, H * W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z


class FFN(nn.Module):
    """Feed Forward Network Block (pre-norm type)"""

    def __init__(self, dim: int, dim_expansion: int) -> None:
        super().__init__()
        self.dim: int = dim
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(self.dim, dim_expansion * self.dim),
            nn.GELU(),
            nn.Linear(dim_expansion * self.dim, self.dim),
        )

        self.norm_pre: nn.LayerNorm = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        # z: [B, H, W, C]
        z_norm: Tensor = self.norm_pre(z)
        z = self.ffn(z_norm) + z

        return z


class ConvBlock2D(nn.Module):
    """Conv Block."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dim_expansion: int,
    ) -> None:
        super().__init__()

        self.dim: int = dim
        self.kernel_size: int = kernel_size

        self.convs: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                self.dim,
                dim_expansion * self.dim,
                self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim_expansion * self.dim,
                self.dim,
                self.kernel_size,
                padding=self.kernel_size // 2,
            ),
        )

        self.convs_1x: nn.Sequential = nn.Sequential(
            nn.Conv2d(self.dim, dim_expansion * self.dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim_expansion * self.dim, self.dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        # x: [B, C, H, W]
        out: Tensor = self.convs(z) + self.convs_1x(z)

        return out


class GlobalAttnBlock(nn.Module):
    """Global 2D Attentions Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_cross_attn: bool = False,
        use_pe: bool = False,
    ) -> None:
        super().__init__()

        self.self_attn: SelfAttnBlock2D = SelfAttnBlock2D(
            dim=dim,
            num_heads=num_heads,
            dim_expansion=dim_expansion,
            use_pe=use_pe,
        )
        if use_cross_attn:
            self.cross_attn: Optional[CrossAttnBlock2D] = CrossAttnBlock2D(
                dim=dim, num_heads=num_heads, dim_expansion=dim_expansion
            )
            self.ffn_c: FFN = FFN(dim=dim, dim_expansion=dim_expansion)
        else:
            self.cross_attn = None
        self.ffn: FFN = FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        # z: [B, HW, C]
        z = z.permute(0, 2, 3, 1)
        if self.cross_attn is not None:
            z = self.cross_attn(z)
            z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0, 3, 1, 2)

        return z.contiguous()


class BasicAttnBlock(nn.Module):
    """1D Attentions Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_pe: bool = False,
    ) -> None:
        super().__init__()

        self.cross_attn: CrossAttnBlock1D = CrossAttnBlock1D(
            dim=dim, num_heads=num_heads, dim_expansion=dim_expansion
        )

        self.self_attn: SelfAttnBlock1D = SelfAttnBlock1D(
            dim=dim,
            num_heads=num_heads,
            dim_expansion=dim_expansion,
            use_pe=use_pe,
        )
        self.ffn_c: FFN = FFN(dim=dim, dim_expansion=dim_expansion)
        self.ffn: FFN = FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        # z: [B, C, H, W]
        z = z.permute(0, 2, 3, 1)
        z = self.cross_attn(z)
        z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0, 3, 1, 2)
        return z
