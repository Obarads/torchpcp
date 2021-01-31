import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from torchpcp.modules.Layer import Layers
from torchpcp.modules.TransformNet import (
    InputTransformNet, 
    FeatureTransformNet
)

class Conv1DModule(Layers):
    def __init__(self, in_channels, out_channels, 
                 act=nn.ReLU(inplace=True)):
        conv = nn.Conv1d(in_channels, out_channels, 1)
        nn.init.xavier_uniform_(conv.weight)
        norm = nn.BatchNorm1d(out_channels)
        super().__init__(conv, norm, act)

class LinearModule(Layers):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        layer = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(layer.weight)
        norm = nn.BatchNorm1d(out_features)
        super().__init__(layer, norm, act)

def create_ModuleList(num_in_channels, num_out_channel_list):
    layers = nn.ModuleList()
    for num_out_channels in num_out_channel_list:
        layers.append(Conv1DModule(num_in_channels, num_out_channels))
        num_in_channels = num_out_channels
    return layers

class PointNetExtractor(nn.Module):
    """
    PointNet encoder: processing between inputs and Maxpooling.
    Parameters
    ----------
    num_out_channel_list_1: [output_channels]
        outputs of layers between input transform net and feature transform net.
    num_out_channel_list_2: [output_channels]
        outputs of layers between feature transform net and max-pooling.
    use_input_transform: bool
        use input transform net
    use_feature_transform: bool
        use feature transform net
    return_features: [layer numbers]
        number list to get layer outputs
    return_transform_features: [layer numbers]
        number list to get transform net outputs
        (0=input transform net, 1=feature transform net)


    Examples
    --------
    batch_size = 4
    num_points = 64
    channels = 3 # xyz
    encoder = PointNetExtractor([32,64], [128, 1024], True, True,
                                return_features=[0, 2], 
                                return_transform_features=[1])
    inputs = torch.tensor(np.random.rand(batch_size, channels, num_points), 
                          dtype=torch.float32)
    global_features, layer_features, transform_net_features, coord_trans, \
        feat_trans = encoder(inputs)
    
    print(layer_features[0]) # 0 layer features
    print(layer_features[1]) # 2 layer features
    print(transform_net_features[0]) # feature transform net features
    """
    def __init__(self, num_out_channel_list_1, num_out_channel_list_2, 
                 use_input_transform, use_feature_transform, 
                 return_features=[], return_transform_features=[]):
        super(PointNetExtractor, self).__init__()

        if use_input_transform:
            self.input_transform_net = InputTransformNet()

        self.encoder1 = create_ModuleList(3, num_out_channel_list_1)

        if use_feature_transform:
            self.feature_transform_net = FeatureTransformNet(
                k=num_out_channel_list_1[-1])

        self.encoder2 = create_ModuleList(num_out_channel_list_1[-1], 
                                          num_out_channel_list_2)

        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform
        self.return_features = return_features
        self.return_transform_features = return_transform_features
        self.num_global_feature_channels = num_out_channel_list_2[-1]

    def forward(self, x):
        # for return_features
        layer_features = []
        transform_net_features = []
        start_idx = 0

        # transpose xyz
        if self.use_input_transform:
            coord_trans = self.input_transform_net(x)
            x = self.transpose(x, coord_trans)
        else:
            coord_trans = None
        ## get transform net features
        if 0 in self.return_transform_features:
            transform_net_features.append(x)

        # encoder1
        for i in range(len(self.encoder1)):
            x = self.encoder1[i](x)
            # retain features
            if i+start_idx in self.return_features:
                layer_features.append(x)
        start_idx = start_idx + len(self.encoder1)

        # transpose features
        if self.use_feature_transform:
            feat_trans = self.feature_transform_net(x)
            x = self.transpose(x, feat_trans)
            if 1 in self.return_transform_features:
                transform_net_features.append(x)
        else:
            feat_trans = None
        ## retain transform net features
        if 0 in self.return_transform_features:
            transform_net_features.append(x)

        # encoder2
        for i in range(len(self.encoder2)):
            x = self.encoder2[i](x)
            # retain features
            if i+start_idx in self.return_features:
                layer_features.append(x)

        # get a global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.num_global_feature_channels)

        return x, layer_features, transform_net_features, coord_trans, feat_trans

    def transpose(self, x, trans):
        x = torch.transpose(x, 1, 2)
        x = torch.bmm(x, trans)
        x = torch.transpose(x, 1, 2)
        return x

class PointNetClassification(nn.Module):
    def __init__(self, num_classes:int, use_input_transform:bool,
                 use_feature_transform:bool):
        """
        PointNet for classification.
        Parameters
        ----------
        num_classes: int
            number of classes for predictions
        use_input_transform: bool
            use transform module for input point clouds
        use_feature_transform: bool
            use transform module for features
        """

        super(PointNetClassification, self).__init__()

        self.encoder = PointNetExtractor([64,64], [64, 128, 1024],
                                         use_input_transform, 
                                         use_feature_transform,
                                         )
        fc = nn.Linear(256, num_classes)
        nn.init.zeros_(fc.bias)
        nn.init.xavier_uniform_(fc.weight)
        self.decoder = nn.Sequential(
            LinearModule(1024, 512),
            nn.Dropout(p=0.3),
            LinearModule(512, 256),
            nn.Dropout(p=0.3),
            fc
        ) 

        self.num_classes = num_classes
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform
    
    def forward(self, inputs):
        """
        PointNet predicts a class label of inputs.
        Parameters
        ----------
        inputs: torch.tensor
            point cloud (inputs.shape = (batch, channel, point))

        Returns
        -------
        pred_labels:torch.tensor
            prediction labels for point clouds (pred_labels.shape = (batch, class))
        """

        x, _, _, coord_trans, feat_trans = self.encoder(inputs)
        pred_labels = self.decoder(x)

        return pred_labels, coord_trans, feat_trans

class PointNetSemanticSegmentation(nn.Module):
    def __init__(self, num_classes:int, num_points:int, use_input_transform:bool,
                 use_feature_transform:bool):
        """
        PointNet for semantic segmentation.
        Parameters
        ----------
        num_classes: int
            number of classes for predictions
        num_points: int
            number of points for inputs
        use_input_transform: bool
            use transform module for input point clouds
        use_feature_transform: bool
            use transform module for features
        """
        super(PointNetSemanticSegmentation, self).__init__()

        self.encoder = PointNetExtractor([64,64],[64,128,1024],
                                         use_input_transform, 
                                         use_feature_transform,
                                         return_transform_features=[1])

        conv = nn.Conv1d(128, num_classes, 1)
        nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(conv.weight)
        self.decoder = nn.Sequential(
            Conv1DModule(1088, 512),
            Conv1DModule(512, 256),
            Conv1DModule(256, 128),
            Conv1DModule(128, 128),
            conv
        )

        self.num_classes = num_classes
        self.num_points = num_points

    def forward(self, x):
        x, _, transform_net_features, coord_trans, feat_trans = self.encoder(x)
        global_info = x.repeat(self.num_points)
        local_info = transform_net_features[0]
        x = torch.cat([local_info, global_info], dim=1) # num_points*1088
        x = self.decoder(x)
        return x, coord_trans, feat_trans

class PointNetPartSegmentation(nn.Module):
    def __init__(self, num_classes:int, num_parts:int, num_points:int, 
                 use_input_transform:bool, use_feature_transform:bool):
        """
        PointNet for part segmentation.
        Parameters
        ----------
        num_classes: int
            number of classes for predictions
        num_points: int
            number of points for inputs
        use_input_transform: bool
            use transform module for input point clouds
        use_feature_transform: bool
            use transform module for features
        """

        super(PointNetPartSegmentation, self).__init__()

        self.encoder = PointNetExtractor([64, 128, 128], [512, 2048],
                                         use_input_transform, 
                                         use_feature_transform,
                                         return_features=[0,1,2,3,4])

        # classification layers
        fc = nn.Linear(256, num_classes)
        nn.init.zeros_(fc.bias)
        nn.init.xavier_uniform_(fc.weight)
        self.cla_decoder = nn.Sequential(
            LinearModule(2048, 256),
            LinearModule(256, 256),
            nn.Dropout(p=0.3),
            fc
        )

        # segmentation layers
        conv = nn.Conv1d(128, num_parts, 1)
        nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(conv.weight)
        self.seg_decoder = nn.Sequential(
            Conv1DModule(4944, 256),
            # Conv1DModule(3024, 256),
            nn.Dropout(p=0.2),
            Conv1DModule(256, 256),
            nn.Dropout(p=0.2),
            Conv1DModule(256, 128),
            conv
        )

        self.num_classes = num_classes
        self.num_parts = num_parts
        self.num_points = num_points
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform
    
    def forward(self, inputs, labels):
        """
        PointNet predicts a class label of inputs.
        Parameters
        ----------
        inputs: torch.tensor
            point cloud (inputs.shape = (batch, channel, point))

        Returns
        -------
        pred_labels:torch.tensor
            prediction labels for point clouds (pred_labels.shape = (batch, class))
        """

        global_features, layer_features, _, coord_trans, feat_trans = self.encoder(inputs)

        pred_cla = self.cla_decoder(global_features)

        out_max = torch.cat([global_features, labels], dim=1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, inputs.shape[2])
        concatenation_features = torch.cat([expand, *layer_features], dim=1)
        pred_seg = self.seg_decoder(concatenation_features)

        return pred_cla, pred_seg, coord_trans, feat_trans



