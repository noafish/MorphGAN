import torch
import torch.nn as nn
import random
from .base_model import BaseModel
from . import networks
from . import dist_model
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from util import util
import cv2
from options.test_options import TestOptions



class PerceptualLoss(nn.Module):
	def __init__(self, opt, model='net-lin', net='vgg'): # VGG using our perceptually-learned weights (LPIPS metric)
	
		print('Setting up Perceptual loss..')
		self.model = dist_model.DistModel()
		self.model.initialize(opt, model=model, net=net)
		print('Done')

	def forward(self, pred, target, normalize=False, epoch=0):
		"""
		Pred and target are Variables.
		If normalize is on, scales images between [-1, 1]
		Assumes the inputs are in range [0, 1].
		"""
		if normalize:
			target = 2 * target - 1
			pred = 2 * pred - 1

		dist = self.model.forward_pair(target, pred)

		return dist

	def forward_layers(self, pred, target, normalize=False, epoch=0, st=0, en=4):
		"""
		Pred and target are Variables.
		If normalize is on, scales images between [-1, 1]
		Assumes the inputs are in range [0, 1].
		"""
		if normalize:
			target = 2 * target - 1
			pred = 2 * pred - 1

		dist = self.model.forward_pair_by_layers(target, pred, st, en)

		return dist

	def forward_single_layer(self, img, l=4):
		return self.model.forward_single_layer(img, l)



class DmorphModel(BaseModel):
	def name(self):
		return 'DMorphModel'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)

		self.counter = 0

		if self.isTrain:
			self.w_d = opt.w_d
			self.w_g = opt.w_g
			self.w_r = opt.w_r
			self.w_p = opt.w_p

		if opt.gray:
			self.nc = 1
		else:
			self.nc = 3

		self.stn = opt.stn
		self.bsz = opt.batchSize
		self.nintrm = opt.nintrm
		self.save_dir = opt.save_dir
		self.costl = opt.costl

		# ablation:
		self.do_gan = not opt.no_gan
		self.do_recon = not opt.no_recon
		self.do_adjp = not opt.no_adjp
		self.do_endpp = not opt.no_endpp
		self.do_adain = not opt.no_adain

		self.fineSize = opt.fineSize

		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['D', 'G', 'P', 'R', 'T']

		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		if self.isTrain:
			self.model_names = ['G', 'D']
		else:  # during test time, only load Gs
			self.model_names = ['G']


		# load/define networks
		self.netG = networks.define_G(self.nc, self.nc, opt.ngf, self.fineSize, opt.init_type, self.gpu_ids, opt.stn, opt.costl, self.do_adain, opt.epoch_count)

		if self.isTrain:
			self.netD = networks.define_D(self.nc, opt.ndf, self.fineSize, opt.init_type, self.gpu_ids)

		if self.isTrain:
			self.criterionGAN = networks.GANLoss(use_lsgan=False).to(self.device)
			self.criterionPerc = PerceptualLoss(opt, 'net')
			self.criterionMSE = torch.nn.MSELoss()
			self.criterionL1 = torch.nn.L1Loss() 
			
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizers = []
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)
			self.loss_P = 0
			self.loss_D = 0
			self.loss_G = 0
			self.loss_R = 0
			self.loss_T = 0


	def scale_img(self, img, scale):
		sz = img.size()
		
		img2 = torch.FloatTensor(sz[0], sz[1], int(scale*sz[2]), int(scale*sz[3]))
		for i in range(0, sz[0]):
			tmp = util.tensor2im(img[i])
			tmp = cv2.resize(tmp, (0,0), fx=scale, fy=scale)
			iimg2 = util.im2tensor(tmp)
			img2[i] = iimg2

		img2 = im2.to(self.device)

		return img2


	def set_input_test(self, input, num):
		self.A = input[0].unsqueeze(0).to(self.device)
		self.B = input[1].unsqueeze(0).to(self.device)
		self.t = np.linspace(-1, 1, num)
		self.content = self.gen_sequence()
		self.style = self.gen_sequence()


	def set_input(self, input, epoch=1, itr=1, scale=1):
		self.counter = self.counter + 1
		self.epoch = epoch
		self.itr = itr
		self.A = input['A'].to(self.device)
		self.B = input['B'].to(self.device)
		self.reals = input['reals']
		sz = self.reals.size()
		self.reals = self.reals.view(-1, sz[2], sz[3], sz[4])
		
		sz = self.A.size()
		self.bsz = sz[0]

		if scale != 1:
			self.A = self.scale_img(self.A, scale)
			self.B = self.scale_img(self.B, scale)
			self.reals = self.scale_img(self.reals, scale)
		
		self.A_path = input['A_paths'][0]
		self.B_path = input['B_paths'][0]
		self.image_paths = [self.A_path, self.B_path]

		self.t = np.linspace(-1, 1, self.nintrm + 2)

		self.content = self.gen_sequence()
		self.style = self.gen_sequence()


	def set_costl(self, cont, stl):
		self.content = cont
		self.style = stl


	def gen_sequence(self):
		pool = np.linspace(-1, 1, 21)
		chosen = np.random.permutation(19) + 1
		chosen2 = chosen[0:self.nintrm]
		t = pool[np.sort(chosen2)]
		t = np.append(-1, t)
		t = np.append(t, 1)
		return t

	def set_nintrm(self, nintrm):
		self.nintrm = nintrm



	def forward(self):
		sz = self.A.size()

		numt = len(self.t)

		tt = torch.from_numpy(self.t)
		tt = tt.type(torch.FloatTensor)
		contt = torch.from_numpy(self.content)
		contt = contt.type(torch.FloatTensor)
		stylet = torch.from_numpy(self.style)
		stylet = stylet.type(torch.FloatTensor)
		
		tt = tt.to(self.device)
		contt = contt.to(self.device)
		stylet = stylet.to(self.device)

		As = self.A
		Bs = self.B

		intrs, intrsr = self.netG(As, Bs, tt, contt, stylet)

		self.intrs = intrs
		self.intrsr = intrsr
		self.Ast = self.netG.module.A
		self.Bst = self.netG.module.B
		self.tst = self.netG.module.tst
		self.tts = self.netG.module.tts
		if self.costl:
			self.cont = self.netG.module.cont
			self.stl = self.netG.module.stl



	def backward_D(self):
		
		pra1, pra2 = self.netD(self.reals)
		loss_D_real = self.criterionGAN(pra1, True) + self.criterionGAN(pra2, True)
 
		pf1, pf2 = self.netD(self.intrs[self.bsz:-self.bsz].detach())
		loss_D_fake = self.criterionGAN(pf1, False) + self.criterionGAN(pf2, False)

		self.loss_D = self.w_d * (loss_D_real + loss_D_fake)
		self.loss_D.backward()


	def backward_G(self):

		numt = len(self.t)

		tot = 0

		if self.do_gan:
			resd1, resd2 = self.netD(self.intrs[self.bsz:-self.bsz])
			self.loss_G = self.criterionGAN(resd1, True) + self.criterionGAN(resd2, True)
			self.loss_G = self.w_g * self.loss_G
			tot += self.loss_G


		if self.stn:
			self.theta = self.netG.module.theta
			self.itheta = self.netG.module.itheta
			ident = self.netG.module.ident
			sz = self.A.size()
			grid = F.interpolate(self.theta, size=(sz[2], sz[3]), mode='bilinear')
			grid = grid.permute(0, 2, 3, 1)
			self.At = F.grid_sample(self.A, grid, padding_mode='border')
			igrid = F.interpolate(self.itheta, size=(sz[2], sz[3]), mode='bilinear')
			igrid = igrid.permute(0, 2, 3, 1)
			self.Bt = F.grid_sample(self.B, igrid, padding_mode='border')

			loss_theta = self.get_dist_layers(self.At, self.B, 4, 4)
			loss_theta = torch.max(loss_theta)
			loss_theta2 = self.criterionMSE(self.theta, ident)

			loss_theta3 = self.get_dist_layers(self.Bt, self.A, 4, 4)
			loss_theta3 = torch.max(loss_theta3)
			loss_theta4 = self.criterionMSE(self.itheta, ident)
			wt_ident = 1
			wt_perc = 0.5
			loss_theta = wt_perc * loss_theta + wt_ident * loss_theta2 + wt_perc * loss_theta3 + wt_ident * loss_theta4
			wt = 1
			self.loss_T = wt * loss_theta
			tot += self.loss_T


		# Reconstruction loss:
		
		if self.do_recon:
			loss_A = self.criterionL1(self.intrs[0:self.bsz], self.A)
			loss_B = self.criterionL1(self.intrs[-self.bsz:], self.B)
			self.loss_R = self.w_r * (loss_A + loss_B) * 0.5
			tot += self.loss_R

		loss_P1 = 0
		loss_P2 = 0
		if self.do_adjp:
			loss_P1 = self.data_adj_loss()
		if self.do_endpp:
			loss_P2 = self.data_endp_loss()
		
		p1w = 1
		p2w = 1
		self.loss_P = p1w * loss_P1 + p2w * loss_P2
		self.loss_P = self.w_p * self.loss_P
		
		tot += self.loss_P

		tot.backward(retain_graph=True)

		self.totloss = tot





	def data_endp_loss(self):
		tst1 = self.tst
		tts1 = self.tts

		if self.costl:
			l = 2
		else:
			l = 3 

		d = self.get_dist_layers(self.intrs, self.Ast, l, l)
		d2 = self.get_dist_layers(self.intrs, self.Bst, l, l)

		if self.costl:
			loss = ((1 - self.stl)*d + self.stl*d2)
		else:
			loss = (tts1*d + tst1*d2)
		loss = torch.mean(loss)
		
		return loss

	def data_adj_loss(self):
		numt = len(self.t)
		
		self.dst = self.get_dist(self.A, self.B)

		if self.costl:
			ccont = self.cont.view(-1, self.bsz)
			ccont = torch.transpose(ccont, 0, 1)
			diff = ccont[:,1:] - ccont[:,0:-1]
			diff2 = diff[:,0:-1] + diff[:,1:]
			diff = diff.contiguous().view(-1)
			diff2 = diff2.contiguous().view(-1)
			mar = self.dst.repeat(numt-1) * diff
			marr = self.dst.repeat(numt-2) * diff2
		else:
			mar0 = self.dst * (1 / (numt-1))
			mar = mar0.repeat(numt-1)
			marr = 2 * mar0.repeat(numt-2)

		a = self.intrs[0:-self.bsz]
		b = self.intrs[self.bsz:]
		a2 = self.intrs[0:-2*self.bsz]
		b2 = self.intrs[2*self.bsz:]
		
		d = self.get_dist(a, b)
		d2 = self.get_dist(a2, b2)

		mloss = (d - mar) ** 2
		mloss2 = (d2 - marr) ** 2

		mloss = mloss.view(-1, self.bsz)
		mloss2 = mloss2.view(-1, self.bsz)

		loss, inds = torch.max(mloss, 0)
		loss2, inds2 = torch.max(mloss2, 0)

		loss = torch.mean(loss)
		loss2 = torch.mean(loss2)
		loss = loss + loss2

		return loss


	def get_dist_layers(self, a, b, st, en):
		d = self.criterionPerc.forward_layers(a, b, st, en)
		return d

	def get_dist(self, a, b):
		return self.perc_dist(a, b)

	def perc_dist(self, a, b):
		d = self.criterionPerc.forward(self.to_rgb(a), self.to_rgb(b), True, self.epoch)
		return d

	def my_mse(self, a, b):
		d = a - b
		d = torch.mean(d ** 2)
		return d


	def to_rgb(self, im):
		if self.nc == 1:
			return im.repeat(1, 3, 1, 1)
		else:
			return im


	def optimize_parameters(self):
		# forward
		self.forward()
		# G
		self.set_requires_grad(self.netD, False)
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
		# D
		if self.do_gan:
			self.set_requires_grad(self.netD, True)
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()



	def calc_deformed(self):

		if self.stn:
			self.theta = self.netG.module.theta
			self.itheta = self.netG.module.itheta
			ident = self.netG.module.ident
			
			sz = self.A.size()
			grid = F.interpolate(self.theta, size=(sz[2], sz[3]), mode='bilinear')
			grid = grid.permute(0, 2, 3, 1)
			self.At = F.grid_sample(self.A, grid, padding_mode='border')
			igrid = F.interpolate(self.itheta, size=(sz[2], sz[3]), mode='bilinear')
			igrid = igrid.permute(0, 2, 3, 1)
			self.Bt = F.grid_sample(self.B, igrid, padding_mode='border')




	def get_linblend(self):
		linblend = self.Ast * self.tts.unsqueeze(1).unsqueeze(2).unsqueeze(3) + self.Bst * self.tst.unsqueeze(1).unsqueeze(2).unsqueeze(3)
		return linblend


	def get_current_visuals(self):
		visual_ret = OrderedDict()
		sp = self.A_path.split('/')
		tp = self.B_path.split('/')
		visual_ret['A'] = [self.A[0], 0, sp[-1]]
		
		visual_ret['B'] = [self.B[0], 1, tp[-1]]

		if self.stn:

			if not self.isTrain:
				self.calc_deformed()

			visual_ret['At'] = [self.At[0], 0, '']
			visual_ret['Bt'] = [self.Bt[0], 1, '']
			linblend = self.Ast * self.tts.unsqueeze(1).unsqueeze(2).unsqueeze(3) + self.Bst * self.tst.unsqueeze(1).unsqueeze(2).unsqueeze(3)

		sz = self.intrs.size()
		seg = np.linspace(0, sz[0]-self.bsz, self.nintrm+2)
		#print(seg)
		seg = [int(i) for i in seg]
		toshow = self.intrs[seg]

		for idx, i in enumerate(toshow):
			
			if self.costl:
				
				name = 'intr_%.2f_%.2f' % (self.content[idx], self.style[idx])
			else:
				name = 'intr_%.2f' % self.t[idx]
			
			visual_ret[name] = [i, self.t[idx], '']

		if self.stn:
			toshow2 = linblend[seg]
			for idx, i in enumerate(toshow2):
				
				if self.costl:
					
					name = 'lblend_%.2f_%.2f' % (self.content[idx], self.style[idx])
				else:
					name = 'lblend_%.2f' % self.t[idx]
				visual_ret[name] = [i, self.t[idx], '']

		return visual_ret
