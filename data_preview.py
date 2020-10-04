import torch 
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

def load():
    train_loader = DataLoader(
                datasets.MNIST("../mnist_data", train=True, download=True,
                transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=30, translate=(0.5,0.5), scale=(0.25, 1),
                        shear = (-30, 30, -30, 30)
                    ),
                    transforms.ToTensor(),
                ])),
                batch_size = 800
    )

    input_batch, labels_batch= next(iter(train_loader))
    grid = utils.make_grid(input_batch, nrow=40, pad_value=1)
    utils.save_image(grid, 'input_batch_preview.png')

if __name__=='__main__':
    load()