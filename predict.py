from turtle import width
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
from torch.utils import data
from models import *

class Predict:

    def __init__(self, sample_frames_path=SAMPLE_FRAME_PATH):
        self.sample_frames_path = sample_frames_path
        self.sample_name = SAMPLE_NAME
        self.classes = CLASSES
        self.index_df = pd.read_csv(SAMPLE_INDEX_DF, index_col=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.params = {'batch_size': BATCH_SIZE, 'shuffle': True,
                       'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        self.cnn_encoder = EncoderCNN()
        self.rnn_decoder = DecoderRNN()

        self.data_loader, self.fnames, self.le = self.prepare_data()

    @staticmethod
    def CRNN_final_prediction(model, device, loader):
        cnn_encoder, rnn_decoder = model
        cnn_encoder.eval()
        rnn_decoder.eval()

        all_y_pred = []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(tqdm(loader)):
                # distribute data to device
                X = X.to(device)
                output = rnn_decoder(cnn_encoder(X))
                # location of max log-probability as prediction
                y_pred = output.max(1, keepdim=True)[1]
                all_y_pred.append(y_pred.cpu().data.squeeze().numpy().tolist())

        return all_y_pred

    def prepare_data(self):

        le, enc = convert2label_and_hot(self.classes)
        actions, all_X_list, fnames = create_X_list_and_actions(
            self.sample_frames_path)
        all_y_list = labels2cat(le, actions)  # all video labels

        transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        selected_frames = np.arange(
            begin_frame, end_frame, skip_frame).tolist()

        # reset data loader
        data_loader = data.DataLoader(Dataset_CRNN(SAMPLE_FRAME_PATH, all_X_list, all_y_list, selected_frames, transform=transform),
                                      **self.params)

        return data_loader, fnames, le

    def predict(self):

        self.cnn_encoder.load_state_dict(torch.load(
            os.path.join(save_model_path, f'cnn_encoder_epoch{EPOCHS}.pth')))

        self.rnn_decoder.load_state_dict(torch.load(
            os.path.join(save_model_path, f'rnn_decoder_epoch{EPOCHS}.pth')))

        print('CRNN model reloaded!')

        all_y_pred = self.CRNN_final_prediction(
            [self.cnn_encoder, self.rnn_decoder], self.device, self.data_loader)

        # Create a dataframe with all the data
        video_df = pd.DataFrame(data={'filename': self.fnames, 'y_pred': cat2labels(
            self.le, all_y_pred)})
        all_data = pd.concat([video_df, self.index_df], axis=1, join='inner')
        only_class1_data = all_data[all_data['y_perd'] == 'class1']

        # Show results - how many class 1 activities
        print('Result: ' + str(all_data['y_pred'].value_counts()))
        all_data.to_csv("././Final_Prediction.csv")
        only_class1_data.to_csv("./Final_Prediction_of_class1.csv")
        print('video prediction finished!')


if __name__ == '__main__':
    p = Predict(SAMPLE_FRAME_PATH)  # Predict on the sample frames
    p.predict()
