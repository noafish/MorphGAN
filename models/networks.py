import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from . import pretrained_networks as pn
from util import util
import numpy as np
import math
import torch.nn.functional as F
import scipy.misc
from . import function


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, sz, init_type='normal', gpu_ids=[], with_stn=False, with_costl=False, with_adain=True, epoch_count=1):

    num = int(math.log(2 * sz, 2)) - 6 + 1
    num = max(num, 2)
    conv_dims = [pow(2, 6+i) for i in range(num)]
    deconv_dims = list(reversed(conv_dims[0:len(conv_dims)]))
    
    netG = MorphGanG(input_nc, output_nc, conv_dims, deconv_dims, gpu_ids, sz, with_stn, with_costl, with_adain)
    inetG = init_net(netG, init_type=init_type, gpu_ids=gpu_ids)
    if with_stn and epoch_count==1:
        netG.init_stn()
    return inetG


def define_D(input_nc, ndf, sz, init_type='normal', gpu_ids=[]):
    output_nc = 1
    hidden_dims = [pow(2, 6 + i) for i in range(int(math.log(2 * sz, 2)) - 6 + 1)]
    netD = MorphGanD(input_nc, output_nc, hidden_dims, gpu_ids)
    return init_net(netD, init_type=init_type, gpu_ids=gpu_ids)



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.L1Loss()


    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def max_se(self, a, b):
        d = (a - b) ** 2
        d = torch.max(d)
        return d

    def my_mse(self, a, b):
        d = a - b
        d = torch.unsqueeze(d, -1)
        d = torch.mean(d ** 2, 1)
        d = torch.mean(d)
        return d

    def __call__(self, input, target_is_real):

        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.my_mse(input, target_tensor)



# Off-the-shelf deep network
class PNet(nn.Module):
    '''Pre-trained network with all channels equally weighted by default'''

    def __init__(self, pnet_type='vgg', pnet_rand=False, use_gpu=True):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1, 3, 1, 1))

        if (self.pnet_type in ['vgg', 'vgg16']):
            self.net = pn.vgg16(pretrained=not self.pnet_rand, requires_grad=False)
        elif (self.pnet_type == 'alex'):
            self.net = pn.alexnet(pretrained=not self.pnet_rand, requires_grad=False)
        elif (self.pnet_type[:-2] == 'resnet'):
            self.net = pn.resnet(pretrained=not self.pnet_rand, requires_grad=False, num=int(self.pnet_type[-2:]))
        elif (self.pnet_type == 'squeeze'):
            self.net = pn.squeezenet(pretrained=not self.pnet_rand, requires_grad=False)

        self.L = self.net.N_slices

        if (use_gpu):
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        if self.pnet_type == 'vggface':
            in0 = (in0 + 1) * 255. / 2
            in0_sc = in0 - self.shift.expand_as(in0)
            in1 = (in1 + 1) * 255. / 2
            in1_sc = in1 - self.shift.expand_as(in1)
        else:
            in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
            in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if (retPerLayer):
            all_scores = []

        
        st = 3
        en = 4

        for i in range(st, en+1):
            cur_score = (1. - util.cos_sim(outs0[i], outs1[i]))
            if (i == st):
                val = 1. * cur_score
            else:
                val = val + cur_score
            if (retPerLayer):
                all_scores += [cur_score]


        if (retPerLayer):
            return (val, all_scores)
        else:
            return val

    def forward_layers(self, in0, in1, st, en):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        for i in range(st, en+1):
            cur_score = (1. - util.cos_sim(outs0[i], outs1[i]))
            if (i == st):
                val = 1. * cur_score
            else:
                val = val + cur_score

        return val

    def forward_single_layers(self, in0, st, en):
        a,b,c,d = in0.size()
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)

        for i in range(st, en+1):
            tmp = outs0[i].view(a,-1)
            if i == st:
                feats = tmp
            else:
                feats = torch.cat((feats, tmp), 1)

        return feats

    def forward_single_layer(self, in0, l):
        in0_sc = (in0 - self.shift.expand_as(in0)) / self.scale.expand_as(in0)

        outs0 = self.net.forward(in0_sc)

        feats = outs0[l]

        return feats




class MorphGanG(nn.Module):

    def __init__(self, input_channel, output_channel, conv_dims, deconv_dims, num_gpu, sz, with_stn, with_costl, with_adain):
        super(MorphGanG, self).__init__()
        self.iscuda = len(num_gpu) > 0
        self.stn = with_stn
        self.costl = with_costl
        self.do_adain = with_adain

        self.grid_sz = 5

        self.layers1 = []
        prev_dim = conv_dims[0]
        self.layers1.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers1.append(nn.LeakyReLU(0.2, inplace=True))

        for out_dim in conv_dims[1:]:
            self.layers1.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers1.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        self.layer_module1 = nn.ModuleList(self.layers1)

        self.layers2 = []
        if self.costl:
            prev_dim = 2 * prev_dim + 3
        else:
            prev_dim = 2 * prev_dim + 1

        for idx, out_dim in enumerate(deconv_dims):
            self.layers2.append(nn.ConvTranspose2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers2.append(nn.LeakyReLU(True))
            prev_dim = out_dim

        self.layers2.append(nn.ConvTranspose2d(prev_dim, output_channel, 1, 1, 0, bias=False))
        self.layers2.append(nn.Tanh())

        self.layer_module2 = nn.ModuleList(self.layers2)

        if self.stn:

            self.layers_stn_loc = []
            self.layers_stn_loc.append(nn.Conv2d(6, 12, kernel_size=7))
            self.layers_stn_loc.append(nn.MaxPool2d(2, stride=2))
            self.layers_stn_loc.append(nn.ReLU(True))
            self.layers_stn_loc.append(nn.Conv2d(12, 24, kernel_size=5))
            self.layers_stn_loc.append(nn.MaxPool2d(2, stride=2))
            self.layers_stn_loc.append(nn.ReLU(True))
            
            self.layer_module_stn_loc = nn.ModuleList(self.layers_stn_loc)

            sz1 = math.floor((sz - 6) / 2)
            sz2 = math.floor((sz1 - 4) / 2)
            
            self.layers_stn_fc = []
            num = 200
            self.layers_stn_fc.append(nn.Linear(24 * sz2 * sz2, num))  # freeform
            self.layers_stn_fc.append(nn.ReLU(True))
            self.layers_stn_fc.append(nn.Linear(num, 2 * self.grid_sz * self.grid_sz))  # freeform
            
            self.layer_module_stn_fc = nn.ModuleList(self.layers_stn_fc)
        

    def init_stn(self):
        self.layer_module_stn_fc[2].weight.data.zero_()
        tmp = np.linspace(-1,1,self.grid_sz)
        tmp = torch.from_numpy(tmp)
        tmp = tmp.repeat(self.grid_sz,1).unsqueeze(0)
        
        tmp2 = tmp.transpose(1,2)
        tmp = torch.cat([tmp, tmp2])
        
        tmp = tmp.float()
        tmp = tmp.view(-1)
        self.layer_module_stn_fc[2].bias.data.copy_(tmp)  # freeform


    def main_stn(self, x, y):
        xy = torch.cat((x,y),1)
        out = xy
        for layer in self.layer_module_stn_loc:
            out = layer(out)

        sz2 = out.size()
        out = out.view(sz2[0], sz2[1] * sz2[2] * sz2[3])
        
        for idx, layer in enumerate(self.layer_module_stn_fc):
            out = layer(out)
            
        return out


    def main_encoder(self, x, y):
        out = x
        for idx, layer in enumerate(self.layer_module1):
            out = layer(out)

        out2 = y
        for idx, layer in enumerate(self.layer_module1):
            out2 = layer(out2)

        return (out, out2)


    def main_decoder(self, stack, t, cont=torch.Tensor(), style=torch.Tensor()):
        numt = t.size()
        numt = numt[0]
        sz = stack.size()
        
        tmp = t.view(numt, 1, 1, 1)
        tmp = tmp.repeat(1, 1, sz[2], sz[3])
        
        if self.costl:
            tmp2 = cont.view(numt, 1, 1, 1)
            tmp3 = style.view(numt, 1, 1, 1)
            
            tmp2 = tmp2.repeat(1, 1, sz[2], sz[3])
            tmp3 = tmp3.repeat(1, 1, sz[2], sz[3])
            
            stack = torch.cat((stack, tmp, tmp2, tmp3), 1)
        else:
            stack = torch.cat((stack, tmp), 1)
        
        out = stack
        for idx, layer in enumerate(self.layer_module2):
            out = layer(out)

        return out


    def b_inv(self, b_mat):
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv


    def get_ident(self, gsz):

        ident = np.linspace(-1,1,gsz[2])
        ident = torch.from_numpy(ident)
        ident = ident.repeat(gsz[2],1).unsqueeze(0)
        
        ident2 = ident.transpose(1,2)
        ident = torch.cat([ident, ident2])
        ident = ident.repeat(gsz[0],1,1,1)
        ident = ident.float()

        if self.iscuda:
            ident = ident.cuda()

        return ident


    def process_t(self, t, sz):

        idx = [i for i in range(t.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        if self.cuda:
            idx = idx.cuda()
        invt = t.index_select(0, idx)

        t2 = (t + 1) / 2
        t2 = t2.repeat(sz, 1)
        tt = t.repeat(sz, 1)
        invtt = invt.repeat(sz, 1)
        t2r = 1 - t2
        t2 = t2.permute(1, 0)
        t2r = t2r.permute(1, 0)
        tt = tt.permute(1, 0)
        invtt = invtt.permute(1, 0)
        t2 = t2.contiguous().view(-1)
        t2r = t2r.contiguous().view(-1)
        tt = tt.contiguous().view(-1)
        invtt = invtt.contiguous().view(-1)

        return (tt, t2, t2r, invtt)


    def main(self, x, y, t, cont, stl):
        numt = t.size()
        numt = numt[0]
        sz0 = x.size()

        tt, t2, t2r, invt = self.process_t(t, sz0[0])

        self.tst = t2
        self.tts = t2r

        if self.costl:
            contt, cont2, cont2r, invcont = self.process_t(cont, sz0[0])
            stlt, stl2, stl2r, invstl = self.process_t(stl, sz0[0])
            self.cont = cont2
            self.stl = stl2

        x2 = x.repeat(numt, 1, 1, 1, 1)
        y2 = y.repeat(numt, 1, 1, 1, 1)
        x22 = x2.view(-1, sz0[1], sz0[2], sz0[3])
        y22 = y2.view(-1, sz0[1], sz0[2], sz0[3])

        self.A = x22
        self.B = y22


        if self.stn:

            xx = x
            yy = y
            theta = self.main_stn(xx, yy)
            itheta  = self.main_stn(yy, xx)

            theta = theta.view(sz0[0], 2, self.grid_sz, self.grid_sz)
            grid = F.interpolate(theta, size=(sz0[2], sz0[3]), mode='bilinear')
            itheta = itheta.view(sz0[0], 2, self.grid_sz, self.grid_sz)
            igrid = F.interpolate(itheta, size=(sz0[2], sz0[3]), mode='bilinear')

            self.theta = theta
            self.itheta = itheta
            
            self.ident = self.get_ident([sz0[0], sz0[1], self.grid_sz, self.grid_sz])

            grid = grid.repeat(numt, 1, 1, 1)
            igrid = igrid.repeat(numt, 1, 1, 1)
            
            ident = self.get_ident(sz0)
            ident = ident.repeat(numt,1,1,1)
            diff_st = grid - ident
            diff_ts = igrid - ident
            
            if self.costl:
                tmp2 = ident + cont2r.unsqueeze(1).unsqueeze(2).unsqueeze(3) * diff_ts
                tmp1 = ident + cont2.unsqueeze(1).unsqueeze(2).unsqueeze(3) * diff_st
            else:
                tmp2 = ident + t2r.unsqueeze(1).unsqueeze(2).unsqueeze(3) * diff_ts
                tmp1 = ident + t2.unsqueeze(1).unsqueeze(2).unsqueeze(3) * diff_st

            tmp1 = tmp1.permute(0, 2, 3, 1)
            tmp2 = tmp2.permute(0, 2, 3, 1)

            A = F.grid_sample(x22, tmp1, padding_mode='border')
            B = F.grid_sample(y22, tmp2, padding_mode='border')

            self.A = A
            self.B = B
            

            out, out2 = self.main_encoder(A, B)
            if self.do_adain:
                if self.costl:
                    out, out2 = function.ada_in(out, out2, stl2)
                else:
                    out, out2 = function.ada_in(out, out2, t2)
            

            ccat = torch.cat((out, out2), 1)
            if self.costl:
                out3 = self.main_decoder(ccat, tt, contt, stlt)
            else:
                out3 = self.main_decoder(ccat, tt)
            ccat2 = torch.cat((out2, out), 1)
            if self.costl:
                out4 = self.main_decoder(ccat2, tt, -1*invcont, -1*invstl)
            else:
                out4 = self.main_decoder(ccat2, tt)

        else:
            
            out, out2 = self.main_encoder(x, y)

            out = out.repeat(numt,1,1,1)
            out2 = out2.repeat(numt,1,1,1)

            out, out2 = function.ada_in(out, out2, t2)

            ccat = torch.cat((out, out2), 1)

            out3 = self.main_decoder(ccat, tt)
            ccat2 = torch.cat((out2, out), 1)

            out4 = self.main_decoder(ccat2, tt)

        return (out3, out4)



    def forward(self, x, y, t, cont=torch.Tensor(), stl=torch.Tensor()):
        return self.main(x, y, t, cont, stl)



class MorphGanD(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_dims, num_gpu):
        super(MorphGanD, self).__init__()
        self.layers = []
        self.layers2 = []

        hidden_dims2 = hidden_dims[:]
        ldim = hidden_dims[-1]
        hidden_dims2 += [2*ldim, 4*ldim]
        
        prev_dim = hidden_dims[0]
        self.layers.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers2.append(nn.Conv2d(input_channel, prev_dim, 4, 2, 1, bias=False))
        self.layers2.append(nn.LeakyReLU(0.2, inplace=True))
        prev_dim2 = prev_dim

        for idx, out_dim in enumerate(hidden_dims[1:]):
            self.layers.append(nn.Conv2d(prev_dim, out_dim, 4, 2, 1, bias=False))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = out_dim

        for idx, out_dim in enumerate(hidden_dims2[1:]):

            self.layers2.append(nn.Conv2d(prev_dim2, out_dim, 4, 2, 1, bias=False))
            self.layers2.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim2 = out_dim

        self.layers.append(nn.Conv2d(prev_dim, output_channel, 1, 1, 0, bias=False))
        self.layers.append(nn.Sigmoid())
        self.layers2.append(nn.Conv2d(prev_dim2, output_channel, 4, 4, 1, bias=False))
        self.layers2.append(nn.Sigmoid())
        
        self.layer_module = nn.ModuleList(self.layers)
        self.layer_module2 = nn.ModuleList(self.layers2)


    def main(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        
        out2 = x
        for layer2 in self.layer_module2:
            out2 = layer2(out2)

        return (out.view(out.size(0), -1), out2.view(out2.size(0), -1))

    def forward(self, x):
        return self.main(x)

