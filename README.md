
# Image Morphing with Perceptual Constraints and STN Alignment

This is a PyTorch implementation of our paper "Image Morphing with Perceptual Constraints and STN Alignment", CGF 2020.



### Abstract

In image morphing, a sequence of plausible frames are synthesized and composited together to form a smooth transformation between given instances. Intermediates must remain faithful to the input, stand on their own as members of the set, and maintain a well-paced visual transition from one to the next. In this paper, we propose a conditional GAN morphing framework operating on a pair of input images. The network is trained to synthesize frames corresponding to temporal samples along the transformation, and learns a proper shape prior that enhances the plausibility of intermediate frames. While individual frame plausibility is boosted by the adversarial setup, a special training protocol producing sequences of frames, combined with a perceptual similarity loss, promote smooth transformation over time. Explicit stating of correspondences is replaced with a grid-based freeform deformation spatial transformer that predicts the geometric warp between the inputs, instituting the smooth geometric effect by bringing the shapes into an initial alignment. We provide comparisons to classic as well as latent space morphing techniques, and demonstrate that, given a set of images for self-supervision, our network learns to generate visually pleasing morphing effects featuring believable in-betweens, with robustness to changes in shape and texture, requiring no correspondence annotation.


### Paper

[arXiv](https://arxiv.org/abs/2004.14071)

[Computer Graphics Forum](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14027)


### Installation

Set up a conda environment, and once in it, install the required Python packages by running:

```
./install_morphgan.sh
```

## Train

Inside the running script run.sh, there are several flags to control the training. See inside the script for details.

For example, to run on the boots dataset, which we have placed in the folder '../dataset/boots', with a resolution of 192, batch size of 1, and a reconstruction weight of 10 (specified inside w.txt), we run:

```
./run.sh -d ../datasets/boots -n morphing -t -s 192 -z 1 -w w.txt
```

The checkpoints of the model, the log and images will be saved to a folder under './checkpoints', which will be given the name DatasetName_ImageSize_GivenNameCounter, where GivenName is specified by the '-n' flag, and counter is a postfix to prevent write overs.

## Test

As before, see inside runt.sh for details pertaining to the flags, but here is an example for running test on the previously specified training session:

```
./runt.sh -n test_morphing -v boots_192_morphing1 -t -s 192
```


## Acknowledgments

This code is heavily based on the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.

## Citation

If you find this code useful, please consider citing our paper:

```
@inproceedings{fish2020image,
	title={Image Morphing With Perceptual Constraints and STN Alignment},
	author={Fish, Noa and Zhang, Richard and Perry, Lilach and Cohen-Or, Daniel and Shechtman, Eli and Barnes, Connelly},
	booktitle={Computer Graphics Forum},
	year={2020},
	organization={Wiley Online Library}
}
```
