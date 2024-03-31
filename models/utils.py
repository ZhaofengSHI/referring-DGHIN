import torch
import torch.nn.functional as F
import torch.nn as nn

def generate_coord(batch, height, width):
    #8-d Coordinate

    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    #print(batch, height, width)
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    #print(batch, height, width)
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)

    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        print("You are using Multi-scale Cross Entropy Loss!")
        # self.weight
        self.loss_weight = [1.0, 0.50, 0.50, 0.50, 0.50]
        self.weight = torch.FloatTensor([0.9, 1.1]).cuda()

    def forward(self, input, target):
        losses = {
            'seg_loss': self.loss_weight[0] * F.cross_entropy(input=input[0], target=target[0], weight=self.weight, reduction='mean'),
            'scale0_loss': self.loss_weight[1] * F.cross_entropy(input=input[1], target=target[1], weight=self.weight, reduction='mean'),
            'scale1_loss': self.loss_weight[2] * F.cross_entropy(input=input[2], target=target[2], weight=self.weight, reduction='mean'),
            'scale2_loss': self.loss_weight[3] * F.cross_entropy(input=input[3], target=target[3], weight=self.weight, reduction='mean'),
            'scale3_loss': self.loss_weight[4] * F.cross_entropy(input=input[4], target=target[4], weight=self.weight, reduction='mean'),
                  }

        return losses

