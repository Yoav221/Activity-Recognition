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


class Trainer:

    def __init__(self, train_data_path=TRAIN_DATA, classes=CLASSES, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, log_interval=LOG_INTERVAL):

        self.classes = classes
        self.train_data_path = train_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.params = {'batch_size': batch_size, 'shuffle': True,
                       'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        self.train_loader, self.valid_loader = self.prepare_data()

        self.cnn_encoder = EncoderCNN()
        self.rnn_decoder = DecoderRNN()

        crnn_params = list(self.cnn_encoder.parameters()) + \
            list(self.rnn_decoder.parameters())
        self.optimizer = torch.optim.Adam(crnn_params, lr=self.learning_rate)

    def prepare_data(self):

        # convert labels -> category
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(CLASSES)
        # convert category -> 1-hot
        action_category = le.transform(CLASSES).reshape(-1, 1)
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(action_category)

        actions = []
        fnames = os.listdir(self.train_data_path)

        all_X_list = []
        for f in fnames:
            loc = f.find('_')
            actions.append(f[: loc])
            all_X_list.append(f)

        all_y_list = labels2cat(le, actions)

        train_list, test_list, train_label, test_label = train_test_split(
            all_X_list, all_y_list, test_size=0.25, random_state=42)

        transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        selected_frames = np.arange(
            begin_frame, end_frame, skip_frame).tolist()

        train_set, valid_set = Dataset_CRNN(self.train_data_path, train_list, train_label, selected_frames, transform=transform), \
            Dataset_CRNN(self.train_data_path, test_list, test_label,
                         selected_frames, transform=transform)

        train_loader = data.DataLoader(
            train_set, **self.params, batch_size=self.batch_size)

        valid_loader = data.DataLoader(
            valid_set, **self.params, batch_size=self.batch_size)

        return train_loader, valid_loader

    def train(self, epochs=EPCOHS):

        # record training process
        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []

        self.cnn_encoder.train()
        self.rnn_decoder.train()

        # start training
        for epoch in range(epochs):
            train_losses, train_scores = train(
                self.log_interval, [self.cnn_encoder, self.rnn_decoder], self.device, self.train_loader, self.optimizer, epoch)
            epoch_test_loss, epoch_test_score = validation(
                [self.cnn_encoder, self.rnn_decoder], self.device, self.optimizer, self.valid_loader, epoch)

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
        # train loss (on epoch end)
        plt.plot(np.arange(1, epochs + 1), A[:, -1])
        plt.plot(np.arange(1, epochs + 1), C)  # test loss (on epoch end)
        plt.title("model loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        # train accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), B[:, -1])
        plt.plot(np.arange(1, epochs + 1), D)  # test accuracy (on epoch end)
        plt.title("training scores")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc="upper left")
        title = "./fig_UCF101_CRNN.png"
        plt.savefig(title, dpi=600)
        # plt.close(fig)
        plt.show()


if __name__ == '__main__':
    t = Trainer()
    t.train(epochs=EPCOHS)
