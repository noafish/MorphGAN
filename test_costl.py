import os
import torch
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import copy
import numpy as np
import time


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    #opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.epoch_count = 2  # not used, just needs to be set
    opt.costl = True
    opt.nintrm = 4
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    torch.manual_seed(1)

    model = create_model(opt)
    model.setup(opt)

    num = opt.nintrm + 2

    x1 = np.linspace(-1, 1, num)
    x2 = -1 * np.ones(num)
    x3 = np.ones(num)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch), True)
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)

        for j in range(num):
            y = 2*(j/(num-1)) - 1
            y2 = y*np.ones(num)
            x = np.linspace(-1, 1, num)
            model.set_costl(x,y2)
            st = time.time()
            model.test()
            en = time.time()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            if i % 5 == 0:
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)


    webpage.save()

