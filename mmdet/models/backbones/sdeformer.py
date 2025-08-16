from mmdet.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader

import torch
import torchinfo
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

def find_points(tensor, type):
    """
    使用不同类型的卷积核检测噪声点
    Args:
        tensor: 输入的脉冲二值图像张量
        type: 使用的卷积核类型（0-6）
    """
    # 根据不同类型选择卷积核
    if type == 1 or type == 0:
        kernel = torch.tensor([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=tensor.dtype)
    elif type == 2:
        kernel = torch.tensor([[2, 0, 2],
                             [0, -8, 0],
                             [2, 0, 2]], dtype=tensor.dtype)
    elif type == 3:
        kernel = torch.tensor([[0, 2, 0],
                             [2, -8, 2],
                             [0, 2, 0]], dtype=tensor.dtype)
    elif type == 4:
        kernel = torch.tensor([[-1,-1,-1,-1,-1],
                             [-1, 1, 1, 1,-1],
                             [-1, 1, 1, 1,-1],
                             [-1, 1, 1, 1,-1],
                             [-1,-1,-1,-1,-1]], dtype=tensor.dtype)
    elif type == 6:  # Sobel风格的边缘感知卷积核
        # 使用Sobel X方向的卷积核
        kernel = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=tensor.dtype)

    # 扩展卷积核维度
    channels = tensor.size(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if type == 4:
        kernel = kernel.expand(channels, 1, 5, 5)
        padding = 2
    else:
        kernel = kernel.expand(channels, 1, 3, 3)
        padding = 1
    kernel = kernel.to(tensor.device)

    # 执行卷积操作
    conv_result = F.conv2d(tensor, kernel, padding=padding, groups=channels).detach()

    # 根据不同类型处理结果
    if type == 1:
        return tensor - (conv_result >= 8).int().detach()
    elif type == 0:
        return tensor + (conv_result <= -6).int().detach()
    elif type == 4:
        return tensor + (conv_result <= -12).int().detach()
    elif type in [2, 3]:
        return tensor - (conv_result <= -8).int().detach()
    elif type == 6:  # 边缘感知卷积核处理
        # 使用阈值3来检测边缘噪声（原阈值6可能过于严格）
        return tensor - (torch.abs(conv_result) >= 5).int().detach()
    
    return tensor

def detect_noise_points(x):
    """
    多阶段去噪处理
    Args:
        x: 输入的脉冲二值图像，支持3D或4D张量
    """
    # 确保输入是4D张量 [B, C, H, W]
    orig_shape = x.shape
    print(f"detect_noise_points input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
    print(f"输入数据统计: 最大值={x.max().item():.4f}, 最小值={x.min().item():.4f}, 平均值={x.mean().item():.4f}, 非零元素比例={torch.count_nonzero(x).item() / x.numel():.4f}")
    
    # 在任何处理之前保存输入的副本
    input_tensor = x.clone()
    
    if x.dim() == 3:
        x = x.unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # 应用多阶段去噪处理
    # 1. 先将图像取反
    x = 1 - x  
    # 2. 应用type 4的降噪（处理大面积噪声）
    # x = find_points(x, 4)
    # 3. 应用type 6的边缘感知降噪
    x = find_points(x, 6)
    # 4. 应用type 3的降噪
    x = find_points(x, 3)
    # 5. 应用type 2的降噪
    x = find_points(x, 2)
    # 6. 再次取反
    x = 1 - x
    # 7. 最后应用type 1的降噪
    x = find_points(x, 1)
    
    # 记录结果并恢复原始维度
    if len(orig_shape) == 3:
        x = x.squeeze(0)
        input_tensor = input_tensor.squeeze(0)
    elif len(orig_shape) == 2:
        x = x.squeeze(0).squeeze(0)
        input_tensor = input_tensor.squeeze(0).squeeze(0)
    
    result = x
    
    # 打印去噪后的统计信息
    print(f"去噪后数据统计: 最大值={result.max().item():.4f}, 最小值={result.min().item():.4f}, 平均值={result.mean().item():.4f}, 非零元素比例={torch.count_nonzero(result).item() / result.numel():.4f}")
    
    # 计算修改的像素点数量
    changed_pixels = torch.count_nonzero(input_tensor != result).item()
    total_pixels = result.numel()
    print(f"去噪修改了 {changed_pixels} 个像素点，占总像素点的 {changed_pixels/total_pixels*100:.2f}%")
        
    return result

  
class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=4):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (i,) = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike_norm4(nn.Module):
    def __init__(
        self,
        Vth=1.0,
        T=4.0,  # 在T上进行Norm
        enable_denoise=True  # 是否启用去噪
    ):
        super().__init__()
        self.spike = Quant()
        self.Vth = Vth
        self.T = T
        self.enable_denoise = enable_denoise

    def forward(self, x):
        if self.training:
            # 脉冲激活
            spike_out = self.spike.apply(x) / self.T
            
            return spike_out
        else:
            return torch.clamp(x, min=0, max=self.T).round_() / self.T


class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        T=None,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike_norm4(T=T)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels),
        )
        self.spike2 = MultiSpike_norm4(T=T)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                med_channels,
                med_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=med_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(med_channels),
        )
        self.spike3 = MultiSpike_norm4(T=T)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.spike1(x)
        # 在第一次脉冲激活后添加去噪，且仅在测试阶段
        # if not self.training:
        #     x = detect_noise_points(x)

        x = self.pwconv1(x)

        x = self.spike2(x)

        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
        return x


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        T=None,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim, T=T)

        self.mlp_ratio = mlp_ratio

        self.spike1 = MultiSpike_norm4(T=T)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = MultiSpike_norm4(T=T)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        # 在spike激活后添加去噪，且仅在测试阶段
        if not self.training:
            x = detect_noise_points(x)
            
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        layer=0,
        T=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike_norm4(T=T)

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike_norm4(T=T)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

        return x


class MS_Attention_linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        T=None,
        lamda_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = MultiSpike_norm4(T=T)

        self.q_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

        self.q_spike = MultiSpike_norm4(T=T)

        self.k_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

        self.k_spike = MultiSpike_norm4(T=T)

        self.v_conv = nn.Sequential(
            nn.Conv2d(dim, int(dim * lamda_ratio), 1, 1, bias=False),
            nn.BatchNorm2d(int(dim * lamda_ratio)),
        )

        self.v_spike = MultiSpike_norm4(T=T)

        self.attn_spike = MultiSpike_norm4(T=T)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim * lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C * self.lamda_ratio)

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale * 2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        T=None,
    ):
        super().__init__()

        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, T=T)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            T=T,
            lamda_ratio=4,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, T=T
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        T=None,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = MultiSpike_norm4(T=T)

    def forward(self, x):
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x


@MODELS.register_module()
class SDEFormer(BaseModule):
    # 添加类变量来追踪批次
    _batch_counter = 0
    _total_images = 0
    
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        self.T = 4
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            T=self.T,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [
                MS_ConvBlock_spike_SepConv(
                    dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios, T=self.T
                )
            ]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [
                MS_ConvBlock_spike_SepConv(
                    dim=embed_dim[0], mlp_ratio=mlp_ratios, T=self.T
                )
            ]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [
                MS_ConvBlock_spike_SepConv(
                    dim=embed_dim[1], mlp_ratio=mlp_ratios, T=self.T
                )
            ]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [
                MS_ConvBlock_spike_SepConv(
                    dim=embed_dim[1], mlp_ratio=mlp_ratios, T=self.T
                )
            ]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            T=self.T,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    T=self.T,
                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            T=self.T,
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    T=self.T,
                )
                for j in range(2)
            ]
        )

        # self.head = (
        #     nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        # )
        # self.spike = MultiSpike_norm4()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                before_state_dict =ckpt['state_dict']
                import collections
                _state_dict = collections.OrderedDict()
                for k,v in before_state_dict.items():
                    _state_dict[k[9:]] = v
            self.load_state_dict(_state_dict, False)

    def forward_features(self, x, hook=None):
        outs = []
        
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)  
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        
        outs.append(x.unsqueeze(0))

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        
        outs.append(x.unsqueeze(0))

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)
        
        outs.append(x.unsqueeze(0))

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        
        outs.append(x.unsqueeze(0))

        return outs

    from mmdet.utils import AvoidCUDAOOM
    @AvoidCUDAOOM.retry_if_cuda_oom
    def forward(self, x):
        import time
        start_time = time.time()
        
        # 更新批次计数器
        SDEFormer._batch_counter += 1
        batch_size = x.shape[0]
        SDEFormer._total_images += batch_size
        
        x = self.forward_features(x)
        torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
        
        # # 打印进度信息
        # print(f"\n{'='*50}")
        # print(f"Batch #{SDEFormer._batch_counter} (处理 {batch_size} 张图像)")
        # print(f"累计处理图像数: {SDEFormer._total_images}")
        # print(f"本批次耗时: {end_time - start_time:.3f}s")
        # print(f"GPU内存占用: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        # print(f"{'='*50}\n")
        
        return x
