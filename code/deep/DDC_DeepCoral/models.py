import torch.nn as nn
from Coral import CORAL
import mmd
import backbone
import torch
import SoS

class Transfer_Net(nn.Module):
    def __init__(self, num_class, mord, base_net='resnet50', transfer_loss='mmd', use_squeeze=False, use_bottleneck=True, bottleneck_width=256, width=1024, gpu_id = 0):
        super(Transfer_Net, self).__init__()
        if transfer_loss=='SoS':
            self.Ms = None
            self.Mt = None
            self.sosloss = SoS.SoS_loss(bottleneck_width, mord = mord, gpu_id=gpu_id) 

        self.gid = gpu_id
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.use_squeeze = use_squeeze 
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(
        ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.LeakyReLU(), nn.Dropout(0.5)] #nn.ReLU()
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target, target_train, source_train, label_source, predictsource=False):
        source = self.base_network(source)
        target = self.base_network(target)
        target_train = self.base_network(target_train)
        source_train = self.base_network(source_train)

        target_clf = self.classifier_layer(target_train)
        if predictsource:
            source_clf = self.classifier_layer(source)
            pred_lbl = torch.cat((source_clf,target_clf),dim=0)
        else:
            pred_lbl = target_clf

        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
            target_train = self.bottleneck_layer(target_train)
            source_train = self.bottleneck_layer(source_train)

        if self.transfer_loss=='SoS' and predictsource==True:
            self.Ms = self.sosloss.mom(source)
            self.Mt = self.sosloss.mom(target)

        if self.transfer_loss!='SoS':
            squeeze_loss = 0
            transfer_loss = self.adapt_loss(source, target, source_train, target_train, label_source, self.transfer_loss)
        else:
            transfer_loss, squeeze_loss = self.adapt_loss(source, target, source_train, target_train, label_source, self.transfer_loss)
            
        return pred_lbl, transfer_loss, squeeze_loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, X_tr, Y_tr, label_source, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral
        
        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral' or 'SoS'

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y, gpu_id=self.gid)
        elif adapt_loss == 'SoS':
            trloss, sqloss = self.sosloss(self.Ms, self.Mt, X, X_tr, Y_tr, label_source, use_squeeze = self.use_squeeze)
            return trloss, sqloss
        else:
            loss = 0
        return loss