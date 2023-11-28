from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer

data_path = 'data/cat.npz'
log_dir = 'log/'
checkpoint_dir = 'checkpoints/'
# data_path = Path.home() / 'MyDatasets/Sketches/apple/train.npy'
# log_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testlogs/3'
# checkpoint_dir = Path.home() / 'MLLogs/SketchRNN/pytorch/apple/testcheckpoints/'

dataset = V5Dataset(str(data_path), 'train', To5vStrokes(), pre_scaling=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=100,shuffle=True)

tb_writer = SummaryWriter(log_dir)
model = SketchRNN()
trainer = Trainer(model, dataloader, tb_writer, checkpoint_dir)

trainer.train(epoch=3000)
tb_writer.close()
