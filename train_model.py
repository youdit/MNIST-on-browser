import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class LitMnsit(pl.LightningModule):

    def __init__(self):
        super(LitMnsit,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.batch_size = 64
        

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x),2)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(),lr=1.0, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1 ,gamma=0.7)
        return [optimizer], [scheduler]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     for parms in self.parameters():
    #         parms.grad = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        J = F.nll_loss(y_hat, y)
        acc = accuracy(y_hat, y)
        # pbar = {'train_acc': acc}
        return {'loss': J}#, 'progress_bar': pbar}
    
    def validation_step(self, batch, batch_idx):
        result = self.training_step(batch, batch_idx)
        # result['progress_bar']['val_acc'] = result['progress_bar']['train_acc']
        # del result['progress_bar']['train_acc']
        # return result
        return result

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        #avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in outputs]).mean()

        #pbar = {'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_val_loss}#, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss= F.nll_loss(y_hat,y, reduce = 'sum')
        pred = y_hat.argmax(dim=1, keepdim=True)
        pred = pred.eq(y.view_as(pred)).sum().item()
        return {'loss': test_loss, "accuracy": pred}
    
    

class MNISTData(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.test_batch_size=1000
    
    def prepare_data(self):
        datasets.MNIST("../mnist_data", train=True, download=True)
        datasets.MNIST("../mnist_data", train=False, download=True)
    
    def setup(self, stage):
        transform = transforms.Compose([
            transforms.RandomAffine(
                degrees =30, translate=(0.5, 0.5), scale=(0.25, 1),
                shear = (-30, 30, -30, 30)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if stage == 'fit':
            mnist_train = datasets.MNIST("../mnist_data", train=True, transform = transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000]) 
        if stage == 'test':
            self.mnist_test = datasets.MNIST("../mnist_data", train=False, transform = transform)

    def train_dataloader(self):
        mnist_train = torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
        return mnist_train
    
    def val_dataloader(self):
        mnist_val = torch.utils.data.DataLoader(self.mnist_val, batch_size = self.batch_size)
        return mnist_val
    
    def test_dataloader(self):
        mnist_test = torch.utils.data.DataLoader(self.mnist_test, batch_size = self.test_batch_size)
        return mnist_test
    
model = LitMnsit()

dm = MNISTData()
trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=10)
trainer.fit(model, dm)
trainer.test(datamodule=dm)
    
trainer.save_checkpoint("model.pt")


