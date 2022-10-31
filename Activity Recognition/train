from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
import sklearn.preprocessing
from torchvision.transforms import transforms
import pandas as pd
from const import *
from functions import *
import matplotlib.pyplot as plt
from models import *
from torch.utils import data

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True,
          'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# convert labels -> category
le = sklearn.preprocessing.LabelEncoder()
le.fit(CLASSES)
# convert category -> 1-hot
action_category = le.transform(CLASSES).reshape(-1, 1)
enc = sklearn.preprocessing.OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(TRAIN_DATA)

all_X_list = []
for f in fnames:
    loc = f.find('_')
    actions.append(f[: loc])
    all_X_list.append(f)

use_cuda, device, params = data_loading_params(batch_size)
# list all data files
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(
    all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(TRAIN_DATA, train_list, train_label, selected_frames, transform=transform), \
    Dataset_CRNN(TRAIN_DATA, test_list, test_label,
                 selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params, batch_size=batch_size)
valid_loader = data.DataLoader(valid_set, **params, batch_size=batch_size)

# Create model
cnn_encoder = EncoderCNN()
rnn_decoder = DecoderRNN()

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

print()
print(os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(1)))

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(
        log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation(
        [cnn_encoder, rnn_decoder], device, optimizer, valid_loader, epoch)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)  # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)  # test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_CRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
