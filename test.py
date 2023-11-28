from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer

data_path = 'data/cat.npz'
log_dir = 'log/'
checkpoint_dir = 'checkpoints/'

dataset = V5Dataset(str(data_path), 'train', To5vStrokes(max_len=200), pre_scaling=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=True)

tb_writer = SummaryWriter(log_dir)
model = SketchRNN(enc_hidden_size=256, dec_hidden_size=512,
                  Nz=128, M=20, dropout=0.1)
trainer = Trainer(model, dataloader, tb_writer,
                  checkpoint_dir, learning_rate=0.0001)

trainer.train(epoch=300)

tb_writer.close()

tb_writer.close()
