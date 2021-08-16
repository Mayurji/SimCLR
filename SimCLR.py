import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

from torch.utils.tensorboard import SummaryWriter

class SimCLR:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)

    def nce_loss(self, features):
        """
        For batch_size=32, each image is transformed into two variant after applying transformation,
        so during each batch of size 32, we have 64 labels.
        features:
            * For batch_size 32, we have 32 x 128, each image is encode into 128 size vector.
            * Similarity matrix, (f, f.T) is (64 x 128, 128 x 64) => (64, 64)

        labels:
            * creates 64 labels, 32 represent one variant of an image and other 32 represents other variant
            of the image.
            * Creating a tensor with diagonal element as 1 and rest as 0 with shape 64x64.
            * Creating a mask using Identity function
            * Substituting negate of the mask over labels, to avoid finding similarity between same variant of 
            the image, thus reducing tensor to 64 x 63
            * Mask removes the main diagonal elements.
        
        Similarity matrix:
            * Mask with main diagonal removed is applied on similarity matrix.
            * similarity matrix represents similarity between each image with other 31 images.
            * NCE loss optimize on these similarity scores by comparing with labels at the end. 
        """
        labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0],-1) #64 x 63
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) #64 x 1
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) #64 x 62
        
        logits = torch.cat([positives, negatives], dim=1) #64 x 63
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device) #64
        logits = logits/ self.args.temperature
        
        return logits, labels

    def train(self, dataloader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0 
        for epoch in range(self.args.epochs):
            for images, _ in dataloader:
                images = torch.cat(images, dim=0).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % 10 == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()
            print(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        print("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        print(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
