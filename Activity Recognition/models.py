import os
from turtle import forward
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from const import *
from PIL import Image
from torch.utils import data


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.fc = nn.Linear(6 * 8 * 8, 48)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            x = self.conv(x_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            cnn_embed_dim.append(x)

        cnn_embed_dim = torch.stack(cnn_embed_dim, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers,
                 h_RNN=RNN_hidden_nodes, h_FC_dim=FC_dim, drop_p=dropout_p, num_classes=2):
        super().__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True)

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)

        x = self.fc1(RNN_out[:, -1, :])
        X = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(
                path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X
