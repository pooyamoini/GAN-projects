
# DCGAN




### What is GAN ?
In generative models, we were always trying to estimate the data distribution by maximum likelihood approach. But in GAN we'll use a whole different approach.
In this approach we have a generative network (as we had before) and we have an extra network called "Discriminator". The discriminator's duty is to recognize fake and real images from each other. The objective for the discriminator is:
$$\text{argmax}_{\theta} \, \mathbb{E}_{x \sim p_{data}} \log[\mathbf{D}_{\theta}(x)] + \mathbb{E}_{x \sim p_{G_{\phi}}} \log[1-\mathbf{D}_{\theta}(x)]$$
And the objective function for generative is:
$$ \text{argmin}_{\phi} \mathbb{E}_{x \sim p_{data}} \log[\mathbf{D}_{\theta}(x)] + \mathbb{E}_{x \sim p_{G_{\phi}}} \log[1-\mathbf{D}_{\theta}(x)] = \text{argmin}_{\phi} \mathbb{E}_{x \sim p_{G_{\phi}}} \log[1-\mathbf{D}_{\theta}(x)]  $$

### Whats is DCGAN?
A DCGAN is a direct extension of the GAN described above, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively.

### DCGAN with celeba dataset


```
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

print('setup done')
```

    setup done



```
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 8

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# During Training
print_every = 100

device = torch.device("cuda:0")
```


```
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive



```
import zipfile
with zipfile.ZipFile("./drive/My Drive/CelebA/Img/img_align_celeba.zip", 'r') as zip_ref:
    zip_ref.extractall("./dataset")
```

#### Creating the dataset


```
dataset_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = dset.ImageFolder(root='./dataset/', transform=dataset_transforms)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```




    <matplotlib.image.AxesImage at 0x7fe0e8f0e630>




![png](DCGAN_celebs%20_files/DCGAN_celebs%20_9_1.png)


#### Weight Initialization

From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. The **weights_init** function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria.


```
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

#### Generator Network

The generator, G, is designed to map the latent space vector (z) to data-space. Since our data are images, converting z to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64). In practice, this is accomplished through a series of strided two dimensional convolutional transpose layers, each paired with a 2d batch norm layer and a relu activation. The output of the generator is fed through a tanh function to return it to the input data range of [−1,1]. It is worth noting the existence of the batch norm functions after the conv-transpose layers, as this is a critical contribution of the DCGAN paper. These layers help with the flow of gradients during training. An image of the generator from the DCGAN paper is shown below.


```
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```

#### Instantiate the generator


```
netG = Generator(ngpu).to(device)

netG.apply(weights_init)

print(netG)
```

    Generator(
      (main): Sequential(
        (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (13): Tanh()
      )
    )


#### Discriminator

As mentioned, the discriminator, $\mathbb D$, is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). Here, $\mathbb D$ takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. This architecture can be extended with more layers if necessary for the problem, but there is significance to the use of the strided convolution, BatchNorm, and LeakyReLUs. The DCGAN paper mentions it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both G and D.


```
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```


```
netD = Discriminator(ngpu).to(device)

netD.apply(weights_init)

print(netD)
```

    Discriminator(
      (main): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (12): Sigmoid()
      )
    )


#### Loss Functions and Optimizers
With D and G setup, we can specify how they learn through the loss functions and optimizers. We will use the Binary Cross Entropy loss (BCELoss) function.Next, we define our real label as 1 and the fake label as 0. These labels will be used when calculating the losses of D and G, and this is also the convention used in the original GAN paper.<br/>
Finally, we set up two separate optimizers, one for D and one for G. As specified in the DCGAN paper, both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5.


```
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

#### Training

##### **Train the Discriminator**
The goal of training the discriminator is to maximize the probability of correctly classifying a given input as real or fake.
Due to the separate mini-batch suggestion from ganhacks, we will calculate this in two steps. First, we will construct a batch of real samples from the training set, forward pass through D, calculate the loss $\log(D(x))$, then calculate the gradients in a backward pass. Secondly, we will construct a batch of fake samples with the current generator, forward pass this batch through D, calculate the loss $\log(1−D(G(z)))$, and accumulate the gradients with a backward pass. Now, with the gradients accumulated from both the all-real and all-fake batches, we call a step of the Discriminator’s optimizer.



##### **Train the Generator**

As stated in the original paper, we want to train the Generator by minimizing $\log(1−D(G(z)))$ in an effort to generate better fakes. As mentioned, this was shown by Goodfellow to not provide sufficient gradients, especially early in the learning process. As a fix, we instead wish to maximize $\log(D(G(z)))$. In the code we accomplish this by: classifying the Generator output from Part 1 with the Discriminator, computing G’s loss using real labels as GT, computing G’s gradients in a backward pass, and finally updating G’s parameters with an optimizer step.


```
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Training has start...")

for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()

        real_data = data[0].to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        # Output training stats
        if i % print_every == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
```

    Training has start...


    /pytorch/aten/src/ATen/native/TensorFactories.cpp:361: UserWarning: Deprecation warning: In a future PyTorch release torch.full will no longer return tensors of floating dtype by default. Instead, a bool fill_value will return a tensor of torch.bool dtype, and an integral fill_value will return a tensor of torch.long dtype. Set the optional `dtype` or `out` arguments to suppress this warning.


    [0/8][0/3166]	Loss_D: 1.8936	Loss_G: 2.2388	D(x): 0.2863	D(G(z)): 0.2728 / 0.1320
    [0/8][100/3166]	Loss_D: 0.0133	Loss_G: 8.5961	D(x): 0.9885	D(G(z)): 0.0004 / 0.0006
    [0/8][200/3166]	Loss_D: 0.4694	Loss_G: 4.7477	D(x): 0.8539	D(G(z)): 0.2000 / 0.0154
    [0/8][300/3166]	Loss_D: 0.2546	Loss_G: 5.6969	D(x): 0.8661	D(G(z)): 0.0716 / 0.0071
    [0/8][400/3166]	Loss_D: 0.4689	Loss_G: 3.6638	D(x): 0.7465	D(G(z)): 0.1026 / 0.0538
    [0/8][500/3166]	Loss_D: 0.3897	Loss_G: 3.9291	D(x): 0.7796	D(G(z)): 0.0385 / 0.0335
    [0/8][600/3166]	Loss_D: 0.4364	Loss_G: 4.9738	D(x): 0.9466	D(G(z)): 0.2656 / 0.0141
    [0/8][700/3166]	Loss_D: 1.3195	Loss_G: 9.2351	D(x): 0.9689	D(G(z)): 0.6303 / 0.0003
    [0/8][800/3166]	Loss_D: 0.9513	Loss_G: 4.1941	D(x): 0.8638	D(G(z)): 0.4203 / 0.0277
    [0/8][900/3166]	Loss_D: 0.3803	Loss_G: 4.6915	D(x): 0.8711	D(G(z)): 0.1756 / 0.0146
    [0/8][1000/3166]	Loss_D: 1.2935	Loss_G: 2.2750	D(x): 0.4329	D(G(z)): 0.0048 / 0.2783
    [0/8][1100/3166]	Loss_D: 0.9997	Loss_G: 5.3613	D(x): 0.8853	D(G(z)): 0.4366 / 0.0130
    [0/8][1200/3166]	Loss_D: 0.2030	Loss_G: 3.5247	D(x): 0.8809	D(G(z)): 0.0384 / 0.0566
    [0/8][1300/3166]	Loss_D: 0.4398	Loss_G: 3.8295	D(x): 0.8511	D(G(z)): 0.1854 / 0.0389
    [0/8][1400/3166]	Loss_D: 0.8163	Loss_G: 3.7583	D(x): 0.7828	D(G(z)): 0.3086 / 0.0402
    [0/8][1500/3166]	Loss_D: 0.7599	Loss_G: 3.6483	D(x): 0.5971	D(G(z)): 0.0489 / 0.0607
    [0/8][1600/3166]	Loss_D: 0.2219	Loss_G: 4.3567	D(x): 0.9606	D(G(z)): 0.1495 / 0.0199
    [0/8][1700/3166]	Loss_D: 0.4567	Loss_G: 3.9849	D(x): 0.8352	D(G(z)): 0.1808 / 0.0319
    [0/8][1800/3166]	Loss_D: 0.3616	Loss_G: 3.7590	D(x): 0.9216	D(G(z)): 0.2157 / 0.0346
    [0/8][1900/3166]	Loss_D: 0.8569	Loss_G: 6.0647	D(x): 0.9663	D(G(z)): 0.4580 / 0.0056
    [0/8][2000/3166]	Loss_D: 0.5795	Loss_G: 6.5512	D(x): 0.8921	D(G(z)): 0.3148 / 0.0029
    [0/8][2100/3166]	Loss_D: 0.4839	Loss_G: 3.0340	D(x): 0.8068	D(G(z)): 0.1773 / 0.0757
    [0/8][2200/3166]	Loss_D: 0.5381	Loss_G: 3.6563	D(x): 0.6789	D(G(z)): 0.0412 / 0.0519
    [0/8][2300/3166]	Loss_D: 0.5467	Loss_G: 1.6393	D(x): 0.6769	D(G(z)): 0.0679 / 0.2557
    [0/8][2400/3166]	Loss_D: 0.2790	Loss_G: 3.7644	D(x): 0.9160	D(G(z)): 0.1549 / 0.0346
    [0/8][2500/3166]	Loss_D: 0.6738	Loss_G: 5.8538	D(x): 0.9223	D(G(z)): 0.3852 / 0.0045
    [0/8][2600/3166]	Loss_D: 0.2253	Loss_G: 4.6969	D(x): 0.8652	D(G(z)): 0.0617 / 0.0160
    [0/8][2700/3166]	Loss_D: 0.4828	Loss_G: 4.8404	D(x): 0.9061	D(G(z)): 0.2708 / 0.0139
    [0/8][2800/3166]	Loss_D: 0.3180	Loss_G: 3.5608	D(x): 0.9054	D(G(z)): 0.1737 / 0.0444
    [0/8][2900/3166]	Loss_D: 0.2148	Loss_G: 4.2644	D(x): 0.9414	D(G(z)): 0.1320 / 0.0215
    [0/8][3000/3166]	Loss_D: 0.5635	Loss_G: 2.5841	D(x): 0.6284	D(G(z)): 0.0340 / 0.1164
    [0/8][3100/3166]	Loss_D: 0.3310	Loss_G: 3.1584	D(x): 0.8708	D(G(z)): 0.1452 / 0.0589
    [1/8][0/3166]	Loss_D: 0.3651	Loss_G: 3.4122	D(x): 0.8886	D(G(z)): 0.1917 / 0.0446
    [1/8][100/3166]	Loss_D: 0.3528	Loss_G: 2.9945	D(x): 0.8328	D(G(z)): 0.1264 / 0.0740
    [1/8][200/3166]	Loss_D: 1.2150	Loss_G: 0.8502	D(x): 0.4032	D(G(z)): 0.0385 / 0.4860
    [1/8][300/3166]	Loss_D: 0.7279	Loss_G: 4.2306	D(x): 0.9159	D(G(z)): 0.4131 / 0.0257
    [1/8][400/3166]	Loss_D: 0.3303	Loss_G: 3.3424	D(x): 0.8439	D(G(z)): 0.1167 / 0.0568
    [1/8][500/3166]	Loss_D: 0.5647	Loss_G: 3.9247	D(x): 0.8698	D(G(z)): 0.2848 / 0.0326
    [1/8][600/3166]	Loss_D: 0.8379	Loss_G: 3.2323	D(x): 0.6787	D(G(z)): 0.2589 / 0.0656
    [1/8][700/3166]	Loss_D: 0.9890	Loss_G: 1.3356	D(x): 0.4470	D(G(z)): 0.0289 / 0.3223
    [1/8][800/3166]	Loss_D: 0.3662	Loss_G: 3.1347	D(x): 0.8473	D(G(z)): 0.1487 / 0.0630
    [1/8][900/3166]	Loss_D: 1.0296	Loss_G: 0.6766	D(x): 0.4605	D(G(z)): 0.0866 / 0.5556
    [1/8][1000/3166]	Loss_D: 0.7136	Loss_G: 1.6014	D(x): 0.6347	D(G(z)): 0.1409 / 0.2473
    [1/8][1100/3166]	Loss_D: 0.4383	Loss_G: 2.2042	D(x): 0.7879	D(G(z)): 0.1408 / 0.1514
    [1/8][1200/3166]	Loss_D: 0.4082	Loss_G: 3.0541	D(x): 0.8319	D(G(z)): 0.1727 / 0.0639
    [1/8][1300/3166]	Loss_D: 1.1033	Loss_G: 3.2204	D(x): 0.7781	D(G(z)): 0.4916 / 0.0574
    [1/8][1400/3166]	Loss_D: 0.3384	Loss_G: 2.8890	D(x): 0.8255	D(G(z)): 0.1190 / 0.0859
    [1/8][1500/3166]	Loss_D: 0.6972	Loss_G: 1.5184	D(x): 0.5775	D(G(z)): 0.0459 / 0.2725
    [1/8][1600/3166]	Loss_D: 0.6277	Loss_G: 2.4635	D(x): 0.6162	D(G(z)): 0.0584 / 0.1336
    [1/8][1700/3166]	Loss_D: 0.6054	Loss_G: 3.2242	D(x): 0.7814	D(G(z)): 0.2463 / 0.0619
    [1/8][1800/3166]	Loss_D: 0.9483	Loss_G: 4.0779	D(x): 0.8868	D(G(z)): 0.5007 / 0.0231
    [1/8][1900/3166]	Loss_D: 0.4893	Loss_G: 2.0601	D(x): 0.7367	D(G(z)): 0.1234 / 0.1619
    [1/8][2000/3166]	Loss_D: 0.6925	Loss_G: 3.6534	D(x): 0.7627	D(G(z)): 0.2585 / 0.0421
    [1/8][2100/3166]	Loss_D: 0.8553	Loss_G: 3.7436	D(x): 0.9051	D(G(z)): 0.4731 / 0.0307
    [1/8][2200/3166]	Loss_D: 0.5517	Loss_G: 2.1849	D(x): 0.7221	D(G(z)): 0.1586 / 0.1475
    [1/8][2300/3166]	Loss_D: 0.5453	Loss_G: 2.9356	D(x): 0.8627	D(G(z)): 0.2915 / 0.0702
    [1/8][2400/3166]	Loss_D: 0.4117	Loss_G: 3.4315	D(x): 0.8467	D(G(z)): 0.1870 / 0.0464
    [1/8][2500/3166]	Loss_D: 0.4734	Loss_G: 3.0367	D(x): 0.7389	D(G(z)): 0.0990 / 0.0674
    [1/8][2600/3166]	Loss_D: 0.4650	Loss_G: 2.8822	D(x): 0.7388	D(G(z)): 0.0798 / 0.0833
    [1/8][2700/3166]	Loss_D: 0.4624	Loss_G: 3.2026	D(x): 0.9139	D(G(z)): 0.2854 / 0.0545
    [1/8][2800/3166]	Loss_D: 1.0038	Loss_G: 4.6143	D(x): 0.8885	D(G(z)): 0.5319 / 0.0141
    [1/8][2900/3166]	Loss_D: 0.4187	Loss_G: 3.6697	D(x): 0.8667	D(G(z)): 0.2165 / 0.0338
    [1/8][3000/3166]	Loss_D: 0.6491	Loss_G: 1.5758	D(x): 0.6101	D(G(z)): 0.0867 / 0.2591
    [1/8][3100/3166]	Loss_D: 0.5037	Loss_G: 3.7268	D(x): 0.9148	D(G(z)): 0.3014 / 0.0351
    [2/8][0/3166]	Loss_D: 0.5570	Loss_G: 2.1456	D(x): 0.6628	D(G(z)): 0.0718 / 0.1570
    [2/8][100/3166]	Loss_D: 0.6439	Loss_G: 2.2065	D(x): 0.6433	D(G(z)): 0.0885 / 0.1381
    [2/8][200/3166]	Loss_D: 0.4985	Loss_G: 1.9267	D(x): 0.7519	D(G(z)): 0.1372 / 0.1760
    [2/8][300/3166]	Loss_D: 0.3076	Loss_G: 3.4268	D(x): 0.8972	D(G(z)): 0.1554 / 0.0422
    [2/8][400/3166]	Loss_D: 0.7646	Loss_G: 0.9378	D(x): 0.5604	D(G(z)): 0.0752 / 0.4642
    [2/8][500/3166]	Loss_D: 1.9657	Loss_G: 6.5946	D(x): 0.9773	D(G(z)): 0.7913 / 0.0030
    [2/8][600/3166]	Loss_D: 0.5828	Loss_G: 4.4481	D(x): 0.9451	D(G(z)): 0.3588 / 0.0165
    [2/8][700/3166]	Loss_D: 0.7063	Loss_G: 1.4389	D(x): 0.6139	D(G(z)): 0.1047 / 0.3041
    [2/8][800/3166]	Loss_D: 1.0816	Loss_G: 6.3416	D(x): 0.9198	D(G(z)): 0.5515 / 0.0030
    [2/8][900/3166]	Loss_D: 0.4571	Loss_G: 3.2939	D(x): 0.8462	D(G(z)): 0.2216 / 0.0487
    [2/8][1000/3166]	Loss_D: 0.7017	Loss_G: 2.2330	D(x): 0.6523	D(G(z)): 0.1530 / 0.1417
    [2/8][1100/3166]	Loss_D: 0.5643	Loss_G: 3.4065	D(x): 0.9024	D(G(z)): 0.3177 / 0.0437
    [2/8][1200/3166]	Loss_D: 0.7173	Loss_G: 4.5141	D(x): 0.9265	D(G(z)): 0.4149 / 0.0177
    [2/8][1300/3166]	Loss_D: 1.6245	Loss_G: 0.4972	D(x): 0.2882	D(G(z)): 0.0361 / 0.6709
    [2/8][1400/3166]	Loss_D: 0.3514	Loss_G: 2.6005	D(x): 0.8757	D(G(z)): 0.1760 / 0.0924
    [2/8][1500/3166]	Loss_D: 0.5645	Loss_G: 2.2068	D(x): 0.7510	D(G(z)): 0.1925 / 0.1397
    [2/8][1600/3166]	Loss_D: 0.4771	Loss_G: 2.6138	D(x): 0.8068	D(G(z)): 0.1807 / 0.1108
    [2/8][1700/3166]	Loss_D: 1.6476	Loss_G: 6.4739	D(x): 0.9359	D(G(z)): 0.7196 / 0.0028
    [2/8][1800/3166]	Loss_D: 0.5249	Loss_G: 3.3952	D(x): 0.9321	D(G(z)): 0.3387 / 0.0415
    [2/8][1900/3166]	Loss_D: 1.9700	Loss_G: 0.2382	D(x): 0.2040	D(G(z)): 0.0227 / 0.8173
    [2/8][2000/3166]	Loss_D: 0.5360	Loss_G: 1.7423	D(x): 0.6608	D(G(z)): 0.0613 / 0.2105
    [2/8][2100/3166]	Loss_D: 0.5401	Loss_G: 2.7396	D(x): 0.8074	D(G(z)): 0.2382 / 0.0837
    [2/8][2200/3166]	Loss_D: 0.4564	Loss_G: 1.9519	D(x): 0.6946	D(G(z)): 0.0499 / 0.1874
    [2/8][2300/3166]	Loss_D: 0.6118	Loss_G: 4.2513	D(x): 0.8955	D(G(z)): 0.3510 / 0.0190
    [2/8][2400/3166]	Loss_D: 0.4687	Loss_G: 2.5184	D(x): 0.7030	D(G(z)): 0.0689 / 0.1134
    [2/8][2500/3166]	Loss_D: 0.7338	Loss_G: 5.4369	D(x): 0.9620	D(G(z)): 0.4444 / 0.0077
    [2/8][2600/3166]	Loss_D: 0.5918	Loss_G: 2.6538	D(x): 0.6450	D(G(z)): 0.0601 / 0.1029
    [2/8][2700/3166]	Loss_D: 0.6399	Loss_G: 3.3253	D(x): 0.8291	D(G(z)): 0.3148 / 0.0540
    [2/8][2800/3166]	Loss_D: 0.3542	Loss_G: 2.9618	D(x): 0.8797	D(G(z)): 0.1792 / 0.0659
    [2/8][2900/3166]	Loss_D: 0.6367	Loss_G: 3.7339	D(x): 0.8227	D(G(z)): 0.3124 / 0.0329
    [2/8][3000/3166]	Loss_D: 0.4842	Loss_G: 2.5943	D(x): 0.7973	D(G(z)): 0.1884 / 0.1054
    [2/8][3100/3166]	Loss_D: 0.5043	Loss_G: 2.0398	D(x): 0.6982	D(G(z)): 0.0895 / 0.1664
    [3/8][0/3166]	Loss_D: 0.5712	Loss_G: 4.9464	D(x): 0.9628	D(G(z)): 0.3430 / 0.0135
    [3/8][100/3166]	Loss_D: 0.4449	Loss_G: 4.0389	D(x): 0.9465	D(G(z)): 0.2888 / 0.0274
    [3/8][200/3166]	Loss_D: 3.1567	Loss_G: 0.6925	D(x): 0.0724	D(G(z)): 0.0020 / 0.5767
    [3/8][300/3166]	Loss_D: 0.4258	Loss_G: 3.3146	D(x): 0.8805	D(G(z)): 0.2251 / 0.0479
    [3/8][400/3166]	Loss_D: 0.8383	Loss_G: 4.9687	D(x): 0.9346	D(G(z)): 0.4676 / 0.0110
    [3/8][500/3166]	Loss_D: 0.6596	Loss_G: 2.6613	D(x): 0.8390	D(G(z)): 0.3233 / 0.0906
    [3/8][600/3166]	Loss_D: 1.8473	Loss_G: 5.6942	D(x): 0.9721	D(G(z)): 0.7548 / 0.0133
    [3/8][700/3166]	Loss_D: 0.4096	Loss_G: 3.4834	D(x): 0.8800	D(G(z)): 0.2162 / 0.0395
    [3/8][800/3166]	Loss_D: 0.3798	Loss_G: 3.6183	D(x): 0.8545	D(G(z)): 0.1793 / 0.0323
    [3/8][900/3166]	Loss_D: 0.5112	Loss_G: 4.0301	D(x): 0.9459	D(G(z)): 0.3230 / 0.0221
    [3/8][1000/3166]	Loss_D: 0.5413	Loss_G: 1.9963	D(x): 0.7078	D(G(z)): 0.1176 / 0.1916
    [3/8][1100/3166]	Loss_D: 0.3537	Loss_G: 3.3008	D(x): 0.8679	D(G(z)): 0.1574 / 0.0505
    [3/8][1200/3166]	Loss_D: 0.7876	Loss_G: 3.9997	D(x): 0.9501	D(G(z)): 0.4297 / 0.0328
    [3/8][1300/3166]	Loss_D: 0.4015	Loss_G: 2.7967	D(x): 0.7518	D(G(z)): 0.0631 / 0.0869
    [3/8][1400/3166]	Loss_D: 0.3985	Loss_G: 3.7893	D(x): 0.8715	D(G(z)): 0.1759 / 0.0359
    [3/8][1500/3166]	Loss_D: 0.4476	Loss_G: 3.5595	D(x): 0.9332	D(G(z)): 0.2749 / 0.0415
    [3/8][1600/3166]	Loss_D: 0.3615	Loss_G: 3.1868	D(x): 0.8189	D(G(z)): 0.1236 / 0.0632
    [3/8][1700/3166]	Loss_D: 0.6426	Loss_G: 1.8920	D(x): 0.5941	D(G(z)): 0.0257 / 0.2244
    [3/8][1800/3166]	Loss_D: 0.6326	Loss_G: 3.9058	D(x): 0.8007	D(G(z)): 0.2769 / 0.0299
    [3/8][1900/3166]	Loss_D: 0.6254	Loss_G: 5.1742	D(x): 0.9488	D(G(z)): 0.3897 / 0.0086
    [3/8][2000/3166]	Loss_D: 0.4965	Loss_G: 1.8463	D(x): 0.7142	D(G(z)): 0.1017 / 0.1936
    [3/8][2100/3166]	Loss_D: 0.5535	Loss_G: 2.3417	D(x): 0.7305	D(G(z)): 0.1609 / 0.1278
    [3/8][2200/3166]	Loss_D: 0.7615	Loss_G: 1.2900	D(x): 0.5681	D(G(z)): 0.0775 / 0.3382
    [3/8][2300/3166]	Loss_D: 0.4665	Loss_G: 2.6126	D(x): 0.7958	D(G(z)): 0.1772 / 0.0966
    [3/8][2400/3166]	Loss_D: 0.4318	Loss_G: 2.7825	D(x): 0.7997	D(G(z)): 0.1531 / 0.0817
    [3/8][2500/3166]	Loss_D: 0.3898	Loss_G: 2.3283	D(x): 0.7824	D(G(z)): 0.0942 / 0.1289
    [3/8][2600/3166]	Loss_D: 0.5797	Loss_G: 0.8606	D(x): 0.6453	D(G(z)): 0.0658 / 0.4945
    [3/8][2700/3166]	Loss_D: 0.3891	Loss_G: 4.4887	D(x): 0.9276	D(G(z)): 0.2356 / 0.0175
    [3/8][2800/3166]	Loss_D: 0.3283	Loss_G: 3.7784	D(x): 0.9776	D(G(z)): 0.2408 / 0.0304
    [3/8][2900/3166]	Loss_D: 0.5019	Loss_G: 3.4844	D(x): 0.8380	D(G(z)): 0.2251 / 0.0445
    [3/8][3000/3166]	Loss_D: 0.5551	Loss_G: 4.3131	D(x): 0.9228	D(G(z)): 0.3313 / 0.0173
    [3/8][3100/3166]	Loss_D: 0.5539	Loss_G: 4.6562	D(x): 0.9526	D(G(z)): 0.3452 / 0.0133
    [4/8][0/3166]	Loss_D: 0.5759	Loss_G: 3.7634	D(x): 0.8486	D(G(z)): 0.2968 / 0.0318
    [4/8][100/3166]	Loss_D: 0.3517	Loss_G: 2.9545	D(x): 0.8335	D(G(z)): 0.1263 / 0.0800
    [4/8][200/3166]	Loss_D: 0.7168	Loss_G: 1.0444	D(x): 0.6218	D(G(z)): 0.1398 / 0.4089
    [4/8][300/3166]	Loss_D: 0.4674	Loss_G: 2.3815	D(x): 0.7722	D(G(z)): 0.1391 / 0.1285
    [4/8][400/3166]	Loss_D: 0.4681	Loss_G: 2.1658	D(x): 0.7015	D(G(z)): 0.0583 / 0.1733
    [4/8][500/3166]	Loss_D: 0.7612	Loss_G: 3.3545	D(x): 0.7424	D(G(z)): 0.2751 / 0.0562
    [4/8][600/3166]	Loss_D: 0.4024	Loss_G: 2.6784	D(x): 0.7830	D(G(z)): 0.0902 / 0.0942
    [4/8][700/3166]	Loss_D: 0.5656	Loss_G: 1.8062	D(x): 0.6712	D(G(z)): 0.0978 / 0.2137
    [4/8][800/3166]	Loss_D: 0.3725	Loss_G: 3.1747	D(x): 0.8174	D(G(z)): 0.1242 / 0.0560
    [4/8][900/3166]	Loss_D: 0.5767	Loss_G: 1.9295	D(x): 0.6920	D(G(z)): 0.0903 / 0.1888
    [4/8][1000/3166]	Loss_D: 0.4092	Loss_G: 3.7029	D(x): 0.8594	D(G(z)): 0.1885 / 0.0421
    [4/8][1100/3166]	Loss_D: 0.6940	Loss_G: 1.7769	D(x): 0.6105	D(G(z)): 0.1028 / 0.2228
    [4/8][1200/3166]	Loss_D: 0.3515	Loss_G: 2.7270	D(x): 0.7852	D(G(z)): 0.0809 / 0.1038
    [4/8][1300/3166]	Loss_D: 0.4028	Loss_G: 4.8046	D(x): 0.9298	D(G(z)): 0.2411 / 0.0136
    [4/8][1400/3166]	Loss_D: 0.2897	Loss_G: 3.1843	D(x): 0.8851	D(G(z)): 0.1354 / 0.0573
    [4/8][1500/3166]	Loss_D: 0.2302	Loss_G: 4.0315	D(x): 0.9086	D(G(z)): 0.1081 / 0.0286
    [4/8][1600/3166]	Loss_D: 0.3362	Loss_G: 2.9902	D(x): 0.8036	D(G(z)): 0.0777 / 0.0779
    [4/8][1700/3166]	Loss_D: 0.5238	Loss_G: 2.3396	D(x): 0.7225	D(G(z)): 0.1306 / 0.1392
    [4/8][1800/3166]	Loss_D: 0.7460	Loss_G: 1.6930	D(x): 0.5754	D(G(z)): 0.0819 / 0.2472
    [4/8][1900/3166]	Loss_D: 0.8642	Loss_G: 1.7704	D(x): 0.5189	D(G(z)): 0.0768 / 0.2603
    [4/8][2000/3166]	Loss_D: 0.1867	Loss_G: 3.5794	D(x): 0.9262	D(G(z)): 0.0969 / 0.0368
    [4/8][2100/3166]	Loss_D: 0.3684	Loss_G: 3.6292	D(x): 0.9088	D(G(z)): 0.2134 / 0.0325
    [4/8][2200/3166]	Loss_D: 0.4594	Loss_G: 2.4523	D(x): 0.7770	D(G(z)): 0.1355 / 0.1173
    [4/8][2300/3166]	Loss_D: 0.4820	Loss_G: 4.5789	D(x): 0.9099	D(G(z)): 0.2751 / 0.0161
    [4/8][2400/3166]	Loss_D: 0.9139	Loss_G: 1.5900	D(x): 0.5161	D(G(z)): 0.0685 / 0.2702
    [4/8][2500/3166]	Loss_D: 0.3824	Loss_G: 3.8805	D(x): 0.9141	D(G(z)): 0.2243 / 0.0320
    [4/8][2600/3166]	Loss_D: 0.7190	Loss_G: 1.5499	D(x): 0.5938	D(G(z)): 0.0614 / 0.2885
    [4/8][2700/3166]	Loss_D: 1.4607	Loss_G: 6.3366	D(x): 0.9808	D(G(z)): 0.6592 / 0.0038
    [4/8][2800/3166]	Loss_D: 0.3821	Loss_G: 2.1705	D(x): 0.7327	D(G(z)): 0.0337 / 0.1532
    [4/8][2900/3166]	Loss_D: 4.1045	Loss_G: 0.0142	D(x): 0.0629	D(G(z)): 0.0281 / 0.9862
    [4/8][3000/3166]	Loss_D: 1.4492	Loss_G: 0.4310	D(x): 0.3324	D(G(z)): 0.0167 / 0.6825
    [4/8][3100/3166]	Loss_D: 1.9885	Loss_G: 0.6264	D(x): 0.2084	D(G(z)): 0.0150 / 0.5878
    [5/8][0/3166]	Loss_D: 0.2115	Loss_G: 3.6961	D(x): 0.8745	D(G(z)): 0.0646 / 0.0424
    [5/8][100/3166]	Loss_D: 0.3376	Loss_G: 2.6099	D(x): 0.8069	D(G(z)): 0.0841 / 0.0999
    [5/8][200/3166]	Loss_D: 0.6648	Loss_G: 0.9748	D(x): 0.6098	D(G(z)): 0.0373 / 0.4318
    [5/8][300/3166]	Loss_D: 0.3388	Loss_G: 4.0782	D(x): 0.9051	D(G(z)): 0.1749 / 0.0265
    [5/8][400/3166]	Loss_D: 0.3533	Loss_G: 2.2403	D(x): 0.7673	D(G(z)): 0.0488 / 0.1780
    [5/8][500/3166]	Loss_D: 0.2962	Loss_G: 2.4644	D(x): 0.8351	D(G(z)): 0.0774 / 0.1284
    [5/8][600/3166]	Loss_D: 1.2498	Loss_G: 4.8247	D(x): 0.9001	D(G(z)): 0.5791 / 0.0135
    [5/8][700/3166]	Loss_D: 1.4120	Loss_G: 0.9929	D(x): 0.3278	D(G(z)): 0.0054 / 0.4494
    [5/8][800/3166]	Loss_D: 0.3752	Loss_G: 3.8703	D(x): 0.9357	D(G(z)): 0.2261 / 0.0302
    [5/8][900/3166]	Loss_D: 0.3097	Loss_G: 2.7762	D(x): 0.7905	D(G(z)): 0.0439 / 0.0957
    [5/8][1000/3166]	Loss_D: 0.6907	Loss_G: 1.2205	D(x): 0.6133	D(G(z)): 0.1050 / 0.3697
    [5/8][1100/3166]	Loss_D: 0.2892	Loss_G: 2.8332	D(x): 0.8508	D(G(z)): 0.0913 / 0.0855
    [5/8][1200/3166]	Loss_D: 0.5464	Loss_G: 2.3007	D(x): 0.6756	D(G(z)): 0.0829 / 0.1388
    [5/8][1300/3166]	Loss_D: 0.2427	Loss_G: 4.0256	D(x): 0.9334	D(G(z)): 0.1456 / 0.0233
    [5/8][1400/3166]	Loss_D: 0.5169	Loss_G: 4.9107	D(x): 0.9111	D(G(z)): 0.2863 / 0.0125
    [5/8][1500/3166]	Loss_D: 0.2711	Loss_G: 3.5380	D(x): 0.9041	D(G(z)): 0.1310 / 0.0439
    [5/8][1600/3166]	Loss_D: 0.6878	Loss_G: 4.2541	D(x): 0.9281	D(G(z)): 0.3792 / 0.0253
    [5/8][1700/3166]	Loss_D: 0.2905	Loss_G: 3.9173	D(x): 0.7965	D(G(z)): 0.0275 / 0.0340
    [5/8][1800/3166]	Loss_D: 0.4862	Loss_G: 2.9863	D(x): 0.8157	D(G(z)): 0.2007 / 0.0688
    [5/8][1900/3166]	Loss_D: 1.0830	Loss_G: 6.5470	D(x): 0.9695	D(G(z)): 0.5615 / 0.0028
    [5/8][2000/3166]	Loss_D: 0.2641	Loss_G: 2.7454	D(x): 0.9020	D(G(z)): 0.1319 / 0.0849
    [5/8][2100/3166]	Loss_D: 0.1761	Loss_G: 4.5784	D(x): 0.9154	D(G(z)): 0.0723 / 0.0206
    [5/8][2200/3166]	Loss_D: 0.2682	Loss_G: 2.8747	D(x): 0.8875	D(G(z)): 0.1241 / 0.0731
    [5/8][2300/3166]	Loss_D: 0.2258	Loss_G: 4.1894	D(x): 0.9391	D(G(z)): 0.1366 / 0.0213
    [5/8][2400/3166]	Loss_D: 0.3053	Loss_G: 4.1539	D(x): 0.9808	D(G(z)): 0.2222 / 0.0230
    [5/8][2500/3166]	Loss_D: 0.3542	Loss_G: 3.6074	D(x): 0.8712	D(G(z)): 0.1695 / 0.0389
    [5/8][2600/3166]	Loss_D: 0.2881	Loss_G: 3.7987	D(x): 0.8858	D(G(z)): 0.1256 / 0.0366
    [5/8][2700/3166]	Loss_D: 0.2264	Loss_G: 3.3826	D(x): 0.9273	D(G(z)): 0.1280 / 0.0469
    [5/8][2800/3166]	Loss_D: 0.3338	Loss_G: 4.2791	D(x): 0.9303	D(G(z)): 0.2054 / 0.0193
    [5/8][2900/3166]	Loss_D: 0.4295	Loss_G: 5.1989	D(x): 0.9340	D(G(z)): 0.2382 / 0.0095
    [5/8][3000/3166]	Loss_D: 2.0029	Loss_G: 6.0203	D(x): 0.8843	D(G(z)): 0.7321 / 0.0056
    [5/8][3100/3166]	Loss_D: 0.4767	Loss_G: 2.6756	D(x): 0.7152	D(G(z)): 0.0718 / 0.1075
    [6/8][0/3166]	Loss_D: 0.4076	Loss_G: 3.0688	D(x): 0.7308	D(G(z)): 0.0432 / 0.0823
    [6/8][100/3166]	Loss_D: 0.5098	Loss_G: 2.3183	D(x): 0.6526	D(G(z)): 0.0166 / 0.1581
    [6/8][200/3166]	Loss_D: 0.2497	Loss_G: 3.3548	D(x): 0.8594	D(G(z)): 0.0737 / 0.0568
    [6/8][300/3166]	Loss_D: 0.4357	Loss_G: 2.3854	D(x): 0.7400	D(G(z)): 0.0611 / 0.1315
    [6/8][400/3166]	Loss_D: 0.3714	Loss_G: 4.5218	D(x): 0.9211	D(G(z)): 0.2155 / 0.0176
    [6/8][500/3166]	Loss_D: 0.6573	Loss_G: 6.4590	D(x): 0.9665	D(G(z)): 0.3735 / 0.0030
    [6/8][600/3166]	Loss_D: 1.2641	Loss_G: 1.0820	D(x): 0.4115	D(G(z)): 0.0698 / 0.4291
    [6/8][700/3166]	Loss_D: 0.6498	Loss_G: 1.4791	D(x): 0.6059	D(G(z)): 0.0477 / 0.3173
    [6/8][800/3166]	Loss_D: 0.4362	Loss_G: 2.1776	D(x): 0.7475	D(G(z)): 0.0815 / 0.1662
    [6/8][900/3166]	Loss_D: 0.1448	Loss_G: 3.3715	D(x): 0.9361	D(G(z)): 0.0697 / 0.0491
    [6/8][1000/3166]	Loss_D: 0.2341	Loss_G: 4.1779	D(x): 0.8910	D(G(z)): 0.0932 / 0.0239
    [6/8][1100/3166]	Loss_D: 0.6708	Loss_G: 6.2076	D(x): 0.9303	D(G(z)): 0.3922 / 0.0036
    [6/8][1200/3166]	Loss_D: 0.3601	Loss_G: 4.0323	D(x): 0.7978	D(G(z)): 0.0733 / 0.0351
    [6/8][1300/3166]	Loss_D: 0.2541	Loss_G: 4.1570	D(x): 0.9554	D(G(z)): 0.1709 / 0.0231
    [6/8][1400/3166]	Loss_D: 0.0913	Loss_G: 4.4016	D(x): 0.9575	D(G(z)): 0.0444 / 0.0235
    [6/8][1500/3166]	Loss_D: 0.2726	Loss_G: 2.9526	D(x): 0.8458	D(G(z)): 0.0634 / 0.0792
    [6/8][1600/3166]	Loss_D: 0.2664	Loss_G: 3.1513	D(x): 0.8488	D(G(z)): 0.0750 / 0.0651
    [6/8][1700/3166]	Loss_D: 0.2248	Loss_G: 4.3234	D(x): 0.9494	D(G(z)): 0.1408 / 0.0201
    [6/8][1800/3166]	Loss_D: 0.6939	Loss_G: 2.7991	D(x): 0.5666	D(G(z)): 0.0209 / 0.1091
    [6/8][1900/3166]	Loss_D: 0.2239	Loss_G: 4.4327	D(x): 0.9324	D(G(z)): 0.1231 / 0.0205
    [6/8][2000/3166]	Loss_D: 0.1692	Loss_G: 4.4604	D(x): 0.8844	D(G(z)): 0.0363 / 0.0219
    [6/8][2100/3166]	Loss_D: 0.2162	Loss_G: 2.7634	D(x): 0.8776	D(G(z)): 0.0593 / 0.1094
    [6/8][2200/3166]	Loss_D: 0.1055	Loss_G: 4.7292	D(x): 0.9184	D(G(z)): 0.0168 / 0.0153
    [6/8][2300/3166]	Loss_D: 0.0838	Loss_G: 4.2353	D(x): 0.9550	D(G(z)): 0.0336 / 0.0230
    [6/8][2400/3166]	Loss_D: 0.2788	Loss_G: 3.3358	D(x): 0.8891	D(G(z)): 0.1202 / 0.0555
    [6/8][2500/3166]	Loss_D: 0.2893	Loss_G: 2.7871	D(x): 0.8657	D(G(z)): 0.1076 / 0.0853
    [6/8][2600/3166]	Loss_D: 0.4085	Loss_G: 5.7829	D(x): 0.9730	D(G(z)): 0.2747 / 0.0051
    [6/8][2700/3166]	Loss_D: 0.3252	Loss_G: 5.5722	D(x): 0.9416	D(G(z)): 0.1999 / 0.0058
    [6/8][2800/3166]	Loss_D: 0.2420	Loss_G: 3.9747	D(x): 0.9315	D(G(z)): 0.1371 / 0.0282
    [6/8][2900/3166]	Loss_D: 0.4032	Loss_G: 5.7510	D(x): 0.9617	D(G(z)): 0.2702 / 0.0046
    [6/8][3000/3166]	Loss_D: 0.2063	Loss_G: 3.6042	D(x): 0.9407	D(G(z)): 0.1200 / 0.0400
    [6/8][3100/3166]	Loss_D: 0.1373	Loss_G: 3.9740	D(x): 0.9438	D(G(z)): 0.0706 / 0.0296
    [7/8][0/3166]	Loss_D: 0.0954	Loss_G: 4.6262	D(x): 0.9399	D(G(z)): 0.0287 / 0.0200
    [7/8][100/3166]	Loss_D: 0.3071	Loss_G: 2.9192	D(x): 0.7999	D(G(z)): 0.0367 / 0.1117
    [7/8][200/3166]	Loss_D: 0.2974	Loss_G: 4.8132	D(x): 0.9275	D(G(z)): 0.1555 / 0.0136
    [7/8][300/3166]	Loss_D: 0.2243	Loss_G: 4.0277	D(x): 0.9262	D(G(z)): 0.1135 / 0.0277
    [7/8][400/3166]	Loss_D: 0.6627	Loss_G: 2.3516	D(x): 0.5767	D(G(z)): 0.0103 / 0.1552
    [7/8][500/3166]	Loss_D: 0.1874	Loss_G: 3.6334	D(x): 0.9200	D(G(z)): 0.0818 / 0.0438
    [7/8][600/3166]	Loss_D: 0.1212	Loss_G: 5.0813	D(x): 0.9701	D(G(z)): 0.0815 / 0.0110
    [7/8][700/3166]	Loss_D: 0.2754	Loss_G: 5.3924	D(x): 0.9552	D(G(z)): 0.1745 / 0.0086
    [7/8][800/3166]	Loss_D: 0.2074	Loss_G: 5.1036	D(x): 0.9488	D(G(z)): 0.1242 / 0.0098
    [7/8][900/3166]	Loss_D: 0.2042	Loss_G: 3.1452	D(x): 0.8896	D(G(z)): 0.0664 / 0.0753
    [7/8][1000/3166]	Loss_D: 0.2280	Loss_G: 4.0873	D(x): 0.9738	D(G(z)): 0.1631 / 0.0230
    [7/8][1100/3166]	Loss_D: 0.9364	Loss_G: 7.7760	D(x): 0.9907	D(G(z)): 0.5218 / 0.0010
    [7/8][1200/3166]	Loss_D: 1.3498	Loss_G: 4.5861	D(x): 0.9553	D(G(z)): 0.5944 / 0.0316
    [7/8][1300/3166]	Loss_D: 0.1181	Loss_G: 4.1017	D(x): 0.9418	D(G(z)): 0.0495 / 0.0266
    [7/8][1400/3166]	Loss_D: 0.1631	Loss_G: 3.7220	D(x): 0.9015	D(G(z)): 0.0402 / 0.0430
    [7/8][1500/3166]	Loss_D: 0.3227	Loss_G: 3.2570	D(x): 0.8673	D(G(z)): 0.1161 / 0.0636
    [7/8][1600/3166]	Loss_D: 0.1163	Loss_G: 4.8130	D(x): 0.9721	D(G(z)): 0.0786 / 0.0133
    [7/8][1700/3166]	Loss_D: 0.4434	Loss_G: 3.4116	D(x): 0.7259	D(G(z)): 0.0478 / 0.0631
    [7/8][1800/3166]	Loss_D: 0.2999	Loss_G: 2.8661	D(x): 0.8080	D(G(z)): 0.0479 / 0.0895
    [7/8][1900/3166]	Loss_D: 0.2273	Loss_G: 2.7873	D(x): 0.8595	D(G(z)): 0.0427 / 0.1115
    [7/8][2000/3166]	Loss_D: 0.0678	Loss_G: 4.9847	D(x): 0.9805	D(G(z)): 0.0454 / 0.0113
    [7/8][2100/3166]	Loss_D: 1.9291	Loss_G: 6.9054	D(x): 0.9799	D(G(z)): 0.7731 / 0.0026
    [7/8][2200/3166]	Loss_D: 0.2833	Loss_G: 2.2991	D(x): 0.8544	D(G(z)): 0.0871 / 0.1487
    [7/8][2300/3166]	Loss_D: 0.1668	Loss_G: 4.2559	D(x): 0.9511	D(G(z)): 0.0873 / 0.0244
    [7/8][2400/3166]	Loss_D: 0.2391	Loss_G: 3.2212	D(x): 0.8306	D(G(z)): 0.0309 / 0.0714
    [7/8][2500/3166]	Loss_D: 0.2725	Loss_G: 3.4166	D(x): 0.8926	D(G(z)): 0.0991 / 0.0523
    [7/8][2600/3166]	Loss_D: 0.0685	Loss_G: 4.4940	D(x): 0.9746	D(G(z)): 0.0404 / 0.0202
    [7/8][2700/3166]	Loss_D: 0.4052	Loss_G: 5.9562	D(x): 0.9908	D(G(z)): 0.2822 / 0.0049
    [7/8][2800/3166]	Loss_D: 0.2882	Loss_G: 4.6913	D(x): 0.8552	D(G(z)): 0.0917 / 0.0201
    [7/8][2900/3166]	Loss_D: 0.3084	Loss_G: 2.6694	D(x): 0.8255	D(G(z)): 0.0585 / 0.1121
    [7/8][3000/3166]	Loss_D: 0.0983	Loss_G: 4.1443	D(x): 0.9541	D(G(z)): 0.0461 / 0.0257
    [7/8][3100/3166]	Loss_D: 0.0793	Loss_G: 5.3754	D(x): 0.9851	D(G(z)): 0.0596 / 0.0068



```
! mkdir ./model
! mkdir ./model/G
! mkdir ./model/D

! touch ./model/G/model.dms
! touch ./model/D/model.dms
```

    mkdir: cannot create directory ‘./model’: File exists
    mkdir: cannot create directory ‘./model/G’: File exists
    mkdir: cannot create directory ‘./model/D’: File exists



```
torch.save({
    'model_state_dict': netG.state_dict(),
}, './model/G/model.dms')

torch.save({
    'model_state_dict': netD.state_dict(),
}, './model/D/model.dms')
```


```
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
```


![png](DCGAN_celebs%20_files/DCGAN_celebs%20_28_0.png)



```
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
```


![png](DCGAN_celebs%20_files/DCGAN_celebs%20_29_0.png)

