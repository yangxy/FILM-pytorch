import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

def warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    assert x.size()[-2:] == flow.size()[-2:]
    flow = flow.permute(0, 2, 3, 1)
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).to(x.dtype)
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output

class SubTreeExtractor(nn.Module):
    def __init__(self, filters=64, sub_levels=4):
        super(SubTreeExtractor, self).__init__()
        k = filters
        n = sub_levels
        in_ch = 3

        self.convs = []
        for i in range(n):
            #conv1 = nn.Sequential(nn.Conv2d(in_ch, k << i, 3, 1, padding='same'), nn.LeakyReLU(negative_slope=0.2)) # padding: SAME
            #conv2 = nn.Sequential(nn.Conv2d(k << i, k << i, 3, 1, padding='same'), nn.LeakyReLU(negative_slope=0.2)) # padding: SAME
            conv1 = nn.Sequential(nn.Conv2d(in_ch, k << i, 3, 1, padding=1), nn.LeakyReLU(negative_slope=0.2))
            conv2 = nn.Sequential(nn.Conv2d(k << i, k << i, 3, 1, padding=1), nn.LeakyReLU(negative_slope=0.2))
            self.convs.append(conv1)
            self.convs.append(conv2)
            in_ch = k << i
        self.convs = nn.ModuleList(self.convs)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # padding: VALID

    def forward(self, image, n):
        head = image
        pyramid = []
        for i in range(n):
            #print(next(self.convs[2*i].parameters()).is_cuda)
            head = self.convs[2*i](head)
            head = self.convs[2*i+1](head)
            pyramid.append(head)
            if i < n-1:
                head = self.pool(head)
        return pyramid

class FeatureExtractor(nn.Module):
    def __init__(self, filters=64, sub_levels=4):
        super(FeatureExtractor, self).__init__()
        self.extract_sublevels = SubTreeExtractor(filters, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid):
        sub_pyramids = []
        for i in range(len(image_pyramid)):
            capped_sub_levels = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(
                self.extract_sublevels(image_pyramid[i], capped_sub_levels))

        feature_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid

class FlowEstimator(nn.Module):
    def __init__(self, in_ch, num_convs, num_filters):
        super(FlowEstimator, self).__init__()

        self.convs = []
        for i in range(num_convs):
            #conv = nn.Sequential(nn.Conv2d(in_ch, num_filters, 3, 1, padding='same'), nn.LeakyReLU(negative_slope=0.2)) # padding: SAME
            conv = nn.Sequential(nn.Conv2d(in_ch, num_filters, 3, 1, padding=1), nn.LeakyReLU(negative_slope=0.2))
            self.convs.append(conv)
            in_ch = num_filters
        #conv = nn.Sequential(nn.Conv2d(num_filters, num_filters//2, 1, 1, padding='same'), nn.LeakyReLU(negative_slope=0.2)) # padding: SAME
        conv = nn.Sequential(nn.Conv2d(num_filters, num_filters//2, 1, 1, padding=0), nn.LeakyReLU(negative_slope=0.2))
        self.convs.append(conv)
        #convs.append(nn.Conv2d(num_filters//2, 2, 1, 1, padding='same')) # padding: SAME
        self.convs.append(nn.Conv2d(num_filters//2, 2, 1, 1, padding=0))
        
        #self.convs = nn.ModuleList(self.convs)
        self.convs = nn.Sequential(*self.convs)

    def forward(self, features_a, features_b):
        net = torch.cat([features_a, features_b], dim=1)
        net = self.convs(net)
        return net

class PyramidFlowEstimator(nn.Module):
    def __init__(self, pyramid_levels=7, specialized_levels=3, in_ch=[128, 384, 896, 1920], flow_convs=[3,3,3,3], flow_filters=[32, 64, 128, 256]):
        super(PyramidFlowEstimator, self).__init__()
        self._predictors = []
        for i in range(specialized_levels):
            self._predictors.append(
                FlowEstimator(in_ch=in_ch[i], num_convs=flow_convs[i], num_filters=flow_filters[i]))

        shared_predictor = FlowEstimator(
        	in_ch=in_ch[-1],
            num_convs=flow_convs[-1],
            num_filters=flow_filters[-1])
        for i in range(specialized_levels, pyramid_levels):
            self._predictors.append(shared_predictor)

        self._predictors = nn.ModuleList(self._predictors)

    def forward(self, feature_pyramid_a, feature_pyramid_b):
        levels = len(feature_pyramid_a)
        v = self._predictors[-1](feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in reversed(range(0, levels-1)):
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2*v, size=level_size)
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = self._predictors[i](feature_pyramid_a[i], warped)
            residuals.append(v_residual)
            v = v_residual + v
        # Use reversed() to return in the 'standard' finest-first-order:
        return list(reversed(residuals))

class Fusion(nn.Module):
    def __init__(self, in_chs, fusion_pyramid_levels=5, specialized_levels=3, filters=64):
        super(Fusion, self).__init__()
        self.levels = fusion_pyramid_levels
        _NUMBER_OF_COLOR_CHANNELS = 3
        self.convs = []
        #in_ch = [128, 256, 512, 1930]
        in_ch = [(filters << i) if i < specialized_levels else (filters << specialized_levels) for i in range(1, fusion_pyramid_levels - 1)]
        in_ch.append(in_chs[-1])

        for i in range(fusion_pyramid_levels - 1):
            m = specialized_levels
            k = filters
            num_filters = (k << i) if i < m else (k << m)

            convs = []
            convs.append(
                #nn.Conv2d(in_ch[i], num_filters, 2, 1, padding='same') # padding: SAME
                nn.Conv2d(in_ch[i], num_filters, 3, 1, padding=1)
            )
            convs.append(
                nn.Sequential(
                    #nn.Conv2d(num_filters+in_chs[i], num_filters, 3, 1, padding='same'), # padding: SAME
                    nn.Conv2d(num_filters+in_chs[i], num_filters, 3, 1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            convs.append(
                nn.Sequential(
                    #nn.Conv2d(num_filters, num_filters, 3, 1, padding='same'), # padding: SAME
                    nn.Conv2d(num_filters, num_filters, 3, 1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2)
                )
            )
            convs = nn.ModuleList(convs)
            self.convs.append(convs)
        self.convs = nn.ModuleList(self.convs)

        # The final convolution that outputs RGB
        #self.output_conv = nn.Conv2d(filters, _NUMBER_OF_COLOR_CHANNELS, 1, padding='valid')
        self.output_conv = nn.Conv2d(filters, _NUMBER_OF_COLOR_CHANNELS, 1, padding=0)

    def forward(self, pyramid):
        if len(pyramid) != self.levels:
            raise ValueError(
                'Fusion called with different number of pyramid levels '
                f'{len(pyramid)} than it was configured for, {self.levels}.')

        net = pyramid[-1]
        for i in reversed(range(0, self.levels - 1)):
            # Resize the tensor from coarser level to match for concatenation.
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, level_size, mode='nearest')
            net = self.convs[i][0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = self.convs[i][1](net)
            net = self.convs[i][2](net)
        net = self.output_conv(net)
        return net

class FILM(nn.Module):
    def __init__(self, pyramid_levels=7, filters=64, specialized_levels=3, sub_levels=4, fusion_pyramid_levels=5, flow_convs=[3,3,3,3], flow_filters=[32,64,128,256]):
        super(FILM, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.filters = filters
        self.specialized_levels = specialized_levels
        self.sub_levels = sub_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels
        self.flow_convs = flow_convs
        self.flow_filters = flow_filters
        in_ch = [sum([2**i for i in range(l)])*filters*2 for l in range(1, sub_levels+1)]
        in_ch2 = [ch + 2*fusion_pyramid_levels for ch in in_ch]

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # padding: VALID
        self.extract = FeatureExtractor(filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(pyramid_levels, specialized_levels, in_ch, flow_convs, flow_filters)
        self.fuse = Fusion(in_ch2, fusion_pyramid_levels, specialized_levels, filters)

    def build_image_pyramid(self, image):
        pyramid = []
        for i in range(self.pyramid_levels):
            pyramid.append(image)
            if i < self.pyramid_levels-1:
                image = self.pool(image)
        
        return pyramid

    def flow_pyramid_synthesis(self, residual_pyramid):
        flow = residual_pyramid[-1]
        flow_pyramid = [flow]
        for residual_flow in reversed(residual_pyramid[:-1]):
            level_size = residual_flow.shape[2:4]
            flow = F.interpolate(2*flow, size=level_size)
            flow = residual_flow + flow
            flow_pyramid.append(flow)
        # Use reversed() to return in the 'standard' finest-first-order:
        return list(reversed(flow_pyramid))

    def multiply_pyramid(self, pyramid, scalar):
        return [image * scalar for image in pyramid]

    def concatenate_pyramids(self, pyramid1, pyramid2):
        result = []
        for features1, features2 in zip(pyramid1, pyramid2):
            result.append(torch.cat([features1, features2], dim=1))
        return result

    def pyramid_warp(self, feature_pyramid, flow_pyramid):
        warped_feature_pyramid = []
        for features, flow in zip(feature_pyramid, flow_pyramid):
            warped_feature_pyramid.append(warp(features, flow))
        return warped_feature_pyramid

    def forward(self, x0, x1, time):
        x0_decoded = x0
        x1_decoded = x1

        # shuffle images
        image_pyramids = [
            self.build_image_pyramid(x0_decoded),
            self.build_image_pyramid(x1_decoded)
        ]

        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]

        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0], feature_pyramids[1])
        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        forward_flow_pyramid = self.flow_pyramid_synthesis(
            forward_residual_flow_pyramid)[:self.fusion_pyramid_levels]
        backward_flow_pyramid = self.flow_pyramid_synthesis(
            backward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        backward_flow = self.multiply_pyramid(backward_flow_pyramid, time)
        forward_flow = self.multiply_pyramid(forward_flow_pyramid, 1 - time)

        pyramids_to_warp = [
            self.concatenate_pyramids(image_pyramids[0][:self.fusion_pyramid_levels],
                                        feature_pyramids[0][:self.fusion_pyramid_levels]),
            self.concatenate_pyramids(image_pyramids[1][:self.fusion_pyramid_levels],
                                        feature_pyramids[1][:self.fusion_pyramid_levels])
        ]

        forward_warped_pyramid = self.pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = self.pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid = self.concatenate_pyramids(forward_warped_pyramid,
                                                      backward_warped_pyramid)
        aligned_pyramid = self.concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid = self.concatenate_pyramids(aligned_pyramid, forward_flow)

        prediction = self.fuse(aligned_pyramid)
        output_color = prediction[:, :3]

        return output_color

