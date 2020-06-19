
# CycleGAN

## Introduction

This notebook demonstrates unpaired image to image translation using conditional GAN's, as described in <a href="https://arxiv.org/pdf/1703.10593.pdf">Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks </a>, also known as CycleGAN. The paper proposes a method that can capture the characteristics of one image domain and figure out how these characteristics could be translated into another image domain, all in the absence of any paired training examples.<br/>
CycleGAN uses a cycle consistency loss to enable training without the need for paired data. In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain. <br/>
This opens up the possibility to do a lot of interesting tasks like photo-enhancement, image colorization, style transfer, etc. All you need is the source and the target dataset (which is simply a directory of images).

Cycle consistency means the result should be close to the original input. For example, if one translates a sentence from English to French, and then translates it back from French to English, then the resulting sentence should be the same as the original sentence.

In cycle consistency loss,


*   Image $\mathbf{X}$ is passed via generator $\mathbf{G}$ that yields generated image $\hat{\mathbf{Y}}$.
*   Generated image $\hat{\mathbf{Y}}$ is passed via generator $\mathbf{F}$ that yields cycled image $\hat{\mathbf{X}}$.

The process is shown below:


![](https://iili.io/JUHabj.png?style=centerme)


## Implementation


```
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt

import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.autograd import Variable
import torch
import numpy as np


print('setup done')
```

    setup done



```
n_epochs = 20
dataset_name = "monet2photo"
batch_size = 8
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
img_height = img_width = 140
channels = 3
sample_interval = 100
n_residual_blocks = 9
lambda_cyc = 10.0
lambda_id = 5.0
checkpoint_interval = 3

device = torch.device("cuda:0")
```

### Creating Dataset


```
! wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
```

    --2020-06-13 10:17:45--  https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
    Resolving people.eecs.berkeley.edu (people.eecs.berkeley.edu)... 128.32.189.73
    Connecting to people.eecs.berkeley.edu (people.eecs.berkeley.edu)|128.32.189.73|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 305231073 (291M) [application/zip]
    Saving to: ‘monet2photo.zip’
    
    monet2photo.zip     100%[===================>] 291.09M  21.0MB/s    in 15s     
    
    2020-06-13 10:18:01 (19.9 MB/s) - ‘monet2photo.zip’ saved [305231073/305231073]
    



```
import zipfile
with zipfile.ZipFile("monet2photo.zip", 'r') as zip_ref:
    zip_ref.extractall("./data")
```


```
! rm -rf monet2photo.zip
```


```
class ImageDataset(Dataset):
    '''
        creating the dataset class for better performance
    '''
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = ImageDataset.to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = ImageDataset.to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    @staticmethod
    def to_rgb(image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image
```


```
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("./data/monet2photo/", transforms_=transforms_, unaligned=True),
    batch_size=16,
    shuffle=True
)

val_dataloader = DataLoader(
    ImageDataset("./data/monet2photo/", transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=2,
    shuffle=True,
    num_workers=1
)
```


```
real_batch = next(iter(dataloader))
plt.figure(figsize=(20, 20))
plt.axis("off")
plt.title("Monet Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch['A'].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```




    <matplotlib.image.AxesImage at 0x7fd3eb925d68>




![png](CycleGAN_monet_files/CycleGAN_monet_12_1.png)


### Creating Models


```
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
```


```
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)
```


```
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
```


```
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)
```


```
input_shape = (channels, img_height, img_width)

G_AB = GeneratorResNet(input_shape, n_residual_blocks).to(device)
G_BA = GeneratorResNet(input_shape, n_residual_blocks).to(device)
D_A = Discriminator(input_shape).to(device)
D_B = Discriminator(input_shape).to(device)

print(G_AB)
print('--'*30)
print(D_B)
```

    GeneratorResNet(
      (model): Sequential(
        (0): ReflectionPad2d((3, 3, 3, 3))
        (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
        (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (9): ReLU(inplace=True)
        (10): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (11): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (12): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (13): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (14): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (15): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (16): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (17): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (18): ResidualBlock(
          (block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (19): Upsample(scale_factor=2.0, mode=nearest)
        (20): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (22): ReLU(inplace=True)
        (23): Upsample(scale_factor=2.0, mode=nearest)
        (24): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (26): ReLU(inplace=True)
        (27): ReflectionPad2d((3, 3, 3, 3))
        (28): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
        (29): Tanh()
      )
    )
    ------------------------------------------------------------
    Discriminator(
      (model): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
        (12): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
      )
    )



```
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)
```




    Discriminator(
      (model): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
        (12): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
      )
    )



### Training


```
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
```


```
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
```


```
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)
```


```
Tensor = torch.cuda.FloatTensor

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
```


```
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)

    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/monet/%s.png" % (batches_done), normalize=False)
```


```
! mkdir ./images
! mkdir ./images/monet
! mkdir ./saved_models/
! mkdir ./saved_models/monet
```

    mkdir: cannot create directory ‘./images’: File exists
    mkdir: cannot create directory ‘./images/monet’: File exists
    mkdir: cannot create directory ‘./saved_models/’: File exists
    mkdir: cannot create directory ‘./saved_models/monet’: File exists



```
prev_time = time.time()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):

        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 200 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/monet/G_AB_%d.pth" % (epoch))
        torch.save(G_BA.state_dict(), "saved_models/monet/G_BA_%d.pth" % (epoch))
        torch.save(D_A.state_dict(), "saved_models/monet/D_A_%d.pth" % (epoch))
        torch.save(D_B.state_dict(), "saved_models/monet/D_B_%d.pth" % (epoch))
```

    [Epoch 0/20] [Batch 0/393] [D loss: 2.261123] [G loss: 12.331038, adv: 2.946524, cycle: 0.626199, identity: 0.624505] ETA: 2:55:08.824782
    [Epoch 0/20] [Batch 200/393] [D loss: 0.197868] [G loss: 5.203292, adv: 0.471380, cycle: 0.336303, identity: 0.273777] ETA: 2:27:18.376546
    [Epoch 1/20] [Batch 0/393] [D loss: 0.211810] [G loss: 3.458359, adv: 0.497585, cycle: 0.201001, identity: 0.190153] ETA: 2:49:51.223123
    [Epoch 1/20] [Batch 200/393] [D loss: 0.177631] [G loss: 3.771797, adv: 0.511451, cycle: 0.219759, identity: 0.212550] ETA: 2:20:03.946647
    [Epoch 2/20] [Batch 0/393] [D loss: 0.152388] [G loss: 3.819916, adv: 0.668568, cycle: 0.216231, identity: 0.197807] ETA: 2:15:55.838777
    [Epoch 2/20] [Batch 200/393] [D loss: 0.155112] [G loss: 3.465252, adv: 0.358320, cycle: 0.206021, identity: 0.209345] ETA: 2:11:55.024130
    [Epoch 3/20] [Batch 0/393] [D loss: 0.183175] [G loss: 3.902381, adv: 0.730961, cycle: 0.214535, identity: 0.205214] ETA: 2:09:02.276547
    [Epoch 3/20] [Batch 200/393] [D loss: 0.199971] [G loss: 3.623505, adv: 0.426564, cycle: 0.211743, identity: 0.215903] ETA: 2:04:00.639208
    [Epoch 4/20] [Batch 0/393] [D loss: 0.170846] [G loss: 3.392118, adv: 0.548899, cycle: 0.199714, identity: 0.169216] ETA: 2:24:46.225891
    [Epoch 4/20] [Batch 200/393] [D loss: 0.199741] [G loss: 3.399910, adv: 0.373881, cycle: 0.208522, identity: 0.188161] ETA: 1:57:16.307825
    [Epoch 5/20] [Batch 0/393] [D loss: 0.182222] [G loss: 3.670897, adv: 0.306796, cycle: 0.237289, identity: 0.198242] ETA: 1:53:07.552718
    [Epoch 5/20] [Batch 200/393] [D loss: 0.160381] [G loss: 3.427809, adv: 0.428126, cycle: 0.204028, identity: 0.191880] ETA: 1:49:07.763759
    [Epoch 6/20] [Batch 0/393] [D loss: 0.144999] [G loss: 3.882208, adv: 0.735369, cycle: 0.216243, identity: 0.196883] ETA: 1:45:54.147394
    [Epoch 6/20] [Batch 200/393] [D loss: 0.179151] [G loss: 3.133725, adv: 0.496661, cycle: 0.175207, identity: 0.176998] ETA: 1:42:07.742449
    [Epoch 7/20] [Batch 0/393] [D loss: 0.101790] [G loss: 2.856257, adv: 0.644830, cycle: 0.147999, identity: 0.146288] ETA: 1:56:59.126878
    [Epoch 7/20] [Batch 200/393] [D loss: 0.159013] [G loss: 3.518093, adv: 0.561627, cycle: 0.206054, identity: 0.179184] ETA: 1:33:48.499095
    [Epoch 8/20] [Batch 0/393] [D loss: 0.170423] [G loss: 2.971188, adv: 0.492985, cycle: 0.169642, identity: 0.156356] ETA: 1:30:19.169400
    [Epoch 8/20] [Batch 200/393] [D loss: 0.167644] [G loss: 2.929248, adv: 0.334684, cycle: 0.176814, identity: 0.165286] ETA: 1:26:16.481194
    [Epoch 9/20] [Batch 0/393] [D loss: 0.177708] [G loss: 4.022618, adv: 0.404529, cycle: 0.260733, identity: 0.202152] ETA: 1:22:36.120026
    [Epoch 9/20] [Batch 200/393] [D loss: 0.190484] [G loss: 2.610689, adv: 0.362011, cycle: 0.151936, identity: 0.145864] ETA: 1:19:12.927899
    [Epoch 10/20] [Batch 0/393] [D loss: 0.177449] [G loss: 2.748815, adv: 0.312600, cycle: 0.166290, identity: 0.154664] ETA: 1:30:03.519502
    [Epoch 10/20] [Batch 200/393] [D loss: 0.188802] [G loss: 2.961307, adv: 0.513799, cycle: 0.165015, identity: 0.159472] ETA: 1:11:30.146878
    [Epoch 11/20] [Batch 0/393] [D loss: 0.159088] [G loss: 2.881064, adv: 0.491704, cycle: 0.162217, identity: 0.153438] ETA: 1:07:50.781811
    [Epoch 11/20] [Batch 200/393] [D loss: 0.176449] [G loss: 2.962711, adv: 0.460378, cycle: 0.171120, identity: 0.158226] ETA: 1:03:48.150271
    [Epoch 12/20] [Batch 0/393] [D loss: 0.252749] [G loss: 3.348329, adv: 0.993545, cycle: 0.160695, identity: 0.149567] ETA: 1:00:14.818480
    [Epoch 12/20] [Batch 200/393] [D loss: 0.164580] [G loss: 3.010215, adv: 0.361243, cycle: 0.181017, identity: 0.167760] ETA: 0:56:18.740570
    [Epoch 13/20] [Batch 0/393] [D loss: 0.151697] [G loss: 2.486084, adv: 0.414214, cycle: 0.139534, identity: 0.135305] ETA: 1:02:48.880835
    [Epoch 13/20] [Batch 200/393] [D loss: 0.138601] [G loss: 3.101573, adv: 0.630549, cycle: 0.168454, identity: 0.157298] ETA: 0:49:02.098830
    [Epoch 14/20] [Batch 0/393] [D loss: 0.161676] [G loss: 2.698353, adv: 0.388308, cycle: 0.157502, identity: 0.147005] ETA: 0:45:17.249612
    [Epoch 14/20] [Batch 200/393] [D loss: 0.187467] [G loss: 2.813728, adv: 0.674581, cycle: 0.145101, identity: 0.137627] ETA: 0:41:22.714300
    [Epoch 15/20] [Batch 0/393] [D loss: 0.113794] [G loss: 2.902203, adv: 0.615462, cycle: 0.151729, identity: 0.153891] ETA: 0:37:39.336977
    [Epoch 15/20] [Batch 200/393] [D loss: 0.162489] [G loss: 2.431412, adv: 0.320196, cycle: 0.143132, identity: 0.135979] ETA: 0:33:55.664215
    [Epoch 16/20] [Batch 0/393] [D loss: 0.160885] [G loss: 2.568894, adv: 0.340707, cycle: 0.150465, identity: 0.144707] ETA: 0:36:02.647619
    [Epoch 16/20] [Batch 200/393] [D loss: 0.141239] [G loss: 3.137288, adv: 0.653681, cycle: 0.161712, identity: 0.173297] ETA: 0:26:16.669965
    [Epoch 17/20] [Batch 0/393] [D loss: 0.164111] [G loss: 2.485894, adv: 0.331959, cycle: 0.142601, identity: 0.145586] ETA: 0:22:31.840566
    [Epoch 17/20] [Batch 200/393] [D loss: 0.163418] [G loss: 2.696450, adv: 0.535891, cycle: 0.142162, identity: 0.147788] ETA: 0:18:44.447522
    [Epoch 18/20] [Batch 0/393] [D loss: 0.145216] [G loss: 2.588751, adv: 0.554510, cycle: 0.134592, identity: 0.137664] ETA: 0:15:04.938629
    [Epoch 18/20] [Batch 200/393] [D loss: 0.177564] [G loss: 2.850957, adv: 0.648925, cycle: 0.147726, identity: 0.144953] ETA: 0:11:14.289856
    [Epoch 19/20] [Batch 0/393] [D loss: 0.135085] [G loss: 2.646099, adv: 0.439228, cycle: 0.145329, identity: 0.150716] ETA: 0:08:57.985876
    [Epoch 19/20] [Batch 200/393] [D loss: 0.209853] [G loss: 2.614296, adv: 0.632469, cycle: 0.133997, identity: 0.128372] ETA: 0:03:41.682442



```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time


PATH = "./images/monet/{:d}.png"

%matplotlib inline

for i in range(6000, 7900, 100):
    plt.figure(figsize=(20, 20))
    p = PATH.format(i)
    image = mpimg.imread(p) # images are color images
    plt.gca().clear()
    plt.imshow(image);
    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(1.0) # wait one second
```


![png](CycleGAN_monet_files/CycleGAN_monet_28_0.png)

