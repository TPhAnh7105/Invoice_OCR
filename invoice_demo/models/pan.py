"""Model definitions extracted from invoice-ocr-with-metrics.ipynb.
Regenerate with: python tools/export_demo_models.py
PAN and OCR live in separate modules to avoid notebook name collisions.
"""
import os, time
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.hub import load_state_dict_from_url
def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0], box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(sorted_boxes, key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]])
    return sorted_boxes

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

def crop_box(img, boxes, out_folder, sort=True):
    h, w, c = img.shape
    if sort:
        boxes = sort_box(boxes)
    for i, box in enumerate(boxes):
        box_name = os.path.join(out_folder, f'{i}.png')
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
        x1, y1, x2, y2, x3, y3, x4, y4 = (int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4))
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1, x2, x3, x4))
        min_y = max(0, min(y1, y2, y3, y4))
        max_x = min(w, max(x1, x2, x3, x4))
        max_y = min(h, max(y1, y2, y3, y4))
        tw = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        th = int(np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2))
        pt1 = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        pt2 = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        cropped = cv2.warpPerspective(img, matrix, (tw, th))
        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, ' is missing')
    return boxes

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return (c2, c3, c4, c5)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
        print('load pretrained models from imagenet')
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

class FPN(nn.Module):

    def __init__(self, backbone_out_channels, **kwargs):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        result_num = kwargs.get('result_num', 6)
        inplace = True
        conv_out = 256
        self.reduce_conv_c2 = nn.Sequential(nn.Conv2d(backbone_out_channels[0], conv_out, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.reduce_conv_c3 = nn.Sequential(nn.Conv2d(backbone_out_channels[1], conv_out, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.reduce_conv_c4 = nn.Sequential(nn.Conv2d(backbone_out_channels[2], conv_out, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.reduce_conv_c5 = nn.Sequential(nn.Conv2d(backbone_out_channels[3], conv_out, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.smooth_p4 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.smooth_p3 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.smooth_p2 = nn.Sequential(nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.conv = nn.Sequential(nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)
        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear')
        return torch.cat([p2, p3, p4, p5], dim=1)

class FPEM_FFM(nn.Module):

    def __init__(self, backbone_out_channels, **kwargs):
        """
        PANnet
        :param backbone_out_channels: 基础网络输出的维度
        """
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        conv_out = 128
        self.reduce_conv_c2 = nn.Sequential(nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=conv_out, kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c3 = nn.Sequential(nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=conv_out, kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c4 = nn.Sequential(nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=conv_out, kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c5 = nn.Sequential(nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=conv_out, kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))
        self.out_conv = nn.Conv2d(in_channels=conv_out * 4, out_channels=6, kernel_size=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear')
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear')
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        return y

class FPEM(nn.Module):

    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))
        return (c2, c3, c4, c5)

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def decode_clip(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold
    kernel = (preds[1] > threshold) * text
    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue
        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)
        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(d_i))
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue
        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return (label, np.array(bbox_list))

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]}, 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]}, 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]}, 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]}, 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]}, 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]}, 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]}}

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}

class Model(nn.Module):

    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']
        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(segmentation_head_dict)
        backbone_model, backbone_out = (backbone_dict[backbone]['models'], backbone_dict[backbone]['out'])
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def trainable_parameters(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = F.interpolate(segmentation_head_out, size=(H, W), mode='bilinear', align_corners=True)
        return y

def get_model(config):
    return Model(config)

class PAN:

    def __init__(self, config, model_path=None, state_dict=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path is not None or state_dict is not None:
            config.model['pretrained'] = False
        self.net = get_model(config.model)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
        elif state_dict is not None:
            self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img, output_dir: str=None, short_size: int=736, crop_region: bool=False):
        """
            Image needs to be in RGB channel
        """
        ori_img = img.copy()
        h, w = ori_img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        tensor = tf.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            preds, boxes_list = decode_clip(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        if crop_region:
            os.makedirs(output_dir, exist_ok=True)
            boxes_list = crop_box(ori_img, boxes_list, output_dir)
        return (preds, boxes_list, t)
