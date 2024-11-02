import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from timm.models.vision_transformer import vit_base_patch16_224
from torchvision.models import vgg19_bn, VGG19_BN_Weights
class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = Conv2D(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = Conv2D(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = Conv2D(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = Conv2D(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2D(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.conv1 = Conv2D(out_channels * 5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, in_channels=3):
        super().__init__()
        # Load a pre-trained ViT model
        self.vit = vit_base_patch16_224(pretrained=True)
        # self.vit = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Modify the input channels
        if in_channels != 3:
            # Adjust the input projection layer
            self.vit.patch_embed.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            # Initialize new weights
            nn.init.xavier_uniform_(self.vit.patch_embed.proj.weight)
            if self.vit.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.vit.patch_embed.proj.bias)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        # Resize x to match ViT input size
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # Extract features from the ViT encoder
        x = self.vit.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        # Exclude the class token and reshape
        x = x[:, 1:, :].transpose(1, 2)  # [batch_size, embed_dim, num_patches]
        h = w = self.img_size // self.patch_size
        x = x.view(-1, self.embed_dim, h, w)
        return x  # [batch_size, embed_dim, h, w]

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c),
            squeeze_excitation_block(out_c)
        )

    def forward(self, x):
        return self.conv(x)

class encoder1(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        network = vgg19(pretrained=True)
        features = list(network.features)

        # Modify the first convolutional layer to accept 'in_channels' channels
        first_conv_layer = features[0]
        new_first_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias is not None
        )

        # Initialize new_first_layer's weights
        with torch.no_grad():
            if in_channels == 3:
                new_first_layer.weight = first_conv_layer.weight
                if first_conv_layer.bias is not None:
                    new_first_layer.bias = first_conv_layer.bias
            elif in_channels > 3:
                new_first_layer.weight[:, :3, :, :] = first_conv_layer.weight
                # Initialize additional channels
                nn.init.xavier_uniform_(new_first_layer.weight[:, 3:, :, :])
                if first_conv_layer.bias is not None:
                    new_first_layer.bias = first_conv_layer.bias
            else:
                # For less than 3 channels
                new_first_layer.weight = first_conv_layer.weight[:, :in_channels, :, :]
                if first_conv_layer.bias is not None:
                    new_first_layer.bias = first_conv_layer.bias

        features[0] = new_first_layer
        self.features = nn.Sequential(*features)

    def forward(self, x):
        skip_connections = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [4, 9, 18, 27]:
                skip_connections.append(x)
        return x, skip_connections[::-1]  # Reverse to match decoder order

class decoder1(nn.Module):
    def __init__(self, skip_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = conv_block(512 + skip_channels[0], 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = conv_block(256 + skip_channels[1], 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = conv_block(128 + skip_channels[2], 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = conv_block(64 + skip_channels[3], 64)

    def forward(self, x, skips):
        x = self.up1(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv4(x)
        return x

class encoder2(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        x1 = self.conv1(x)
        skip_connections.append(x1)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        skip_connections.append(x2)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        skip_connections.append(x3)
        x = self.pool3(x3)

        x4 = self.conv4(x)
        skip_connections.append(x4)
        x = self.pool4(x4)

        return x, skip_connections[::-1]

class decoder2(nn.Module):
    def __init__(self, skip_channels1, skip_channels2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = conv_block(512 + 512 + 512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = conv_block(256 + 256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = conv_block(128 + 128 + 128, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = conv_block(64 + 64 + 64, 64)

    def forward(self, x, skip1, skip2):
        # First upsampling
        x = self.up1(x)  # x: [B, 512, 32, 32]
        # print(f'x after up1: {x.shape}')
        # print(f'skip1[0] before upsample: {skip1[0].shape}')
        skip1_0 = F.interpolate(skip1[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        # print(f'skip1[0] after upsample: {skip1_0.shape}')
        skip2_0 = skip2[0]  # Already [B, 512, 32, 32]
        x = torch.cat([x, skip1_0, skip2_0], dim=1)
        x = self.conv1(x)
        
        # Second upsampling
        x = self.up2(x)  # x: [B, 256, 64, 64]
        skip1_1 = F.interpolate(skip1[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_1 = skip2[1]
        x = torch.cat([x, skip1_1, skip2_1], dim=1)
        x = self.conv2(x)

        # Third upsampling
        x = self.up3(x)  # x: [B, 128, 128, 128]
        skip1_2 = F.interpolate(skip1[2], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_2 = skip2[2]
        x = torch.cat([x, skip1_2, skip2_2], dim=1)
        x = self.conv3(x)

        # Fourth upsampling
        x = self.up4(x)  # x: [B, 64, 256, 256]
        skip1_3 = F.interpolate(skip1[3], size=x.shape[2:], mode='bilinear', align_corners=False)
        skip2_3 = skip2[3]
        x = torch.cat([x, skip1_3, skip2_3], dim=1)
        x = self.conv4(x)
        return x


class build_doubleunet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.e1 = encoder1(in_channels=in_channels)
        self.a1 = ASPP(512, 512)

        # Integrate ViT block
        self.vit = ViTBlock(img_size=224, patch_size=16, embed_dim=768, in_channels=in_channels)

        # Add projection layers to match channels
        self.vit_proj1 = nn.Conv2d(768, 512, kernel_size=1)
        self.vit_proj2 = nn.Conv2d(768, 512, kernel_size=1)

        self.d1 = None  # Initialize decoder1 later
        self.outc1 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2(in_channels=in_channels)
        self.a2 = ASPP(512, 512)

        self.d2 = None  # Initialize decoder2 later
        self.outc2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # First U-Net
        x1, skip1 = self.e1(x)
        skip_channels1 = [s.size(1) for s in skip1]
        if self.d1 is None:
            self.d1 = decoder1(skip_channels1).to(x.device)
        x1 = self.a1(x1)
        vit_features = self.vit(x)
        vit_features_proj1 = self.vit_proj1(vit_features)
        x1 = x1 + F.interpolate(vit_features_proj1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.d1(x1, skip1)
        y1 = self.outc1(x1)

        # Prepare input for second U-Net
        if self.num_classes == 1:
            y1_sigmoid = self.sigmoid(y1)
        else:
            y1_softmax = torch.softmax(y1, dim=1)
            y1_sigmoid = torch.sum(y1_softmax[:, 1:, :, :], dim=1, keepdim=True)

        # Upsample y1_sigmoid to match x
        y1_sigmoid_upsampled = F.interpolate(y1_sigmoid, size=x.shape[2:], mode='bilinear', align_corners=False)

        x2_input = x * y1_sigmoid_upsampled

        # Second U-Net
        x2, skip2 = self.e2(x2_input)
        skip_channels2 = [s.size(1) for s in skip2]
        if self.d2 is None:
            self.d2 = decoder2(skip_channels1, skip_channels2).to(x.device)
        x2 = self.a2(x2)
        vit_features_proj2 = self.vit_proj2(vit_features)
        x2 = x2 + F.interpolate(vit_features_proj2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = self.d2(x2, skip1, skip2)
        y2 = self.outc2(x2)
        return y1, y2

if __name__ == "__main__":    
    in_channels = 4  # Example for Task01_BrainTumour
    num_classes = 4
    x = torch.randn((2, in_channels, 256, 256))
    model = build_doubleunet(in_channels=in_channels, num_classes=num_classes)
    y1, y2 = model(x)
    print(f'y1 shape: {y1.shape}, y2 shape: {y2.shape}')
