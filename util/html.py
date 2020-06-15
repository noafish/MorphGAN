import dominate
from dominate.tags import *
import os
import imageio
from . import util
import re
from operator import itemgetter


class HTML:
    def __init__(self, web_dir, title, is_test=False, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        self.is_test = is_test

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        if self.is_test:
            with self.doc:
                h3(str)
        else:
            with self.edoc:
                h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        #self.doc.add(self.t)
        if self.is_test:
            self.doc.add(self.t)
        else:
            self.edoc.add(self.t)

    def add_images(self, epoch, ims, txts, links, width=400):
        if epoch != -1:
            img_dir_epoch = os.path.join(self.img_dir, str(epoch))
            util.mkdirs(img_dir_epoch)
        else:
            img_dir_epoch = self.img_dir
        tmp = ims[0]
        tmp2 = tmp.split('_')
        regex = re.compile(".*(intr).*")
        intr_inds = [i for i, j in enumerate(txts) if regex.match(j)]
        if len(intr_inds) > 0:
            ims_for_gif = itemgetter(*intr_inds)(ims)
            gifname0 = '_'.join(tmp2[0:2]) + '.gif'
            gifname = os.path.join(img_dir_epoch, gifname0)
            self.gen_gif(img_dir_epoch, ims_for_gif, gifname)
            ims = [gifname0] + ims
            txts = ['gif'] + txts
            links = [gifname0] + links
        path_parts = img_dir_epoch.split('/')
        if self.is_test:
            rel_path = path_parts[-1:]
        else:
            rel_path = path_parts[-2:]
        rel_path = '/'.join(rel_path)
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join(rel_path, link)):
                                img(style="width:%dpx" % width, src=os.path.join(rel_path, im))
                            br()
                            p(txt)

    
    def put_images(self, ims, txts, links, rel_path, width=400):
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join(rel_path, link)):
                                img(style="width:%dpx" % width, src=os.path.join(rel_path, im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

    def add_epoch_doc(self, epoch):
        ttl = 'epoch %d' % epoch
        self.edoc = dominate.document(title=ttl)

    def add_epoch_link(self, str, epoch):
        with self.doc:
            nm = 'epoch %d' % epoch
            a(nm, href=str, style="font-size:50px")
            br()

    def save_top(self, epoch):
        epoch_file = 'epoch_%d.html' % epoch
        self.add_epoch_link(epoch_file, epoch)
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

    def save_epoch(self, epoch):
        html_file = '%s/epoch_%d.html' % (self.web_dir, epoch)
        f = open(html_file, 'wt')
        f.write(self.edoc.render())
        f.close()

    def gen_gif(self, imdir, ims, outname):

        images = []
        for idx, fname in enumerate(ims):
            im = imageio.imread(os.path.join(imdir, fname))
            images.append(im)

        imageio.mimsave(outname, images, fps=1.5)


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
