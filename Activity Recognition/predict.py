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

'''In this code we create samples video from the video, 
and the model will run on all of them and make a prediction,
wether It's an explosion or not. 
The result will be saved in ./Final_Prediction_bomb_classifier.pkl'''


def main():

    # If the samples don't exist create them, otherwise pass.
    if not os.path.exists(SAMPLE_VIDEO_PATH) or (os.stat(SAMPLE_VIDEO_PATH).st_size == 0):
        print("Creating Sample Videos ----->")
        samples = Sample(main_video_path=INPUT_VID, explosion_name='single_explosion',
                         non_explosion_name='non_explosion', sample_name=SAMPLE_NAME)

        frames_data, width_data, height_data = samples.create_samples(
            FRAME_JUMP, WIDTH_JUMP, HEIGHT_JUMP, DIM_LIMIT)
        print('Samples Videos have been created!')

    else:
        print('Sample videos already exist!')

    if not os.path.exists(SAMPLE_FRAME_PATH) or (os.stat(SAMPLE_FRAME_PATH).st_size == 0):
        print("Creating Sample Frames ----->")
        SaveFrames.save_frames_from_multiple_videos(
            SAMPLE_NAME, SAMPLE_VIDEO_PATH, SAMPLE_FRAME_PATH)
        print('Samples Frames have been created!')

    else:
        print('Sample frames already exist!')

    le, enc = convert2label_and_hot(CLASSES)
    actions, all_X_list, fnames = create_X_list_and_actions(SAMPLE_FRAME_PATH)
    all_y_list = labels2cat(le, actions)  # all video labels

    # data loading parameters
    use_cuda, device, params = data_loading_params(batch_size=batch_size)

    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # reset data loader
    all_data_loader = data.DataLoader(Dataset_CRNN(SAMPLE_FRAME_PATH, all_X_list, all_y_list, selected_frames, transform=transform),
                                      **params)

    # reload CRNN model
    cnn_encoder = EncoderCNN()
    rnn_decoder = DecoderRNN()

    cnn_encoder.load_state_dict(torch.load(
        os.path.join(save_model_path, 'cnn_encoder_epoch4.pth')))
    rnn_decoder.load_state_dict(torch.load(
        os.path.join(save_model_path, 'rnn_decoder_epoch4.pth')))
    print('CRNN model reloaded!')

    # make all video predictions by reloaded model
    print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
    all_y_pred = CRNN_final_prediction(
        [cnn_encoder, rnn_decoder], device, all_data_loader)

    # write in pandas dataframe
    df = pd.DataFrame(data={'filename': fnames, 'y_pred': cat2labels(
        le, all_y_pred), 'frame_index': frames_data, 'width_index': width_data, 'hegith_index': height_data})

    # Show results - how many explosions
    print(df['y_pred'].value_counts())
    df.to_pickle("./Final_Prediction.pkl")  # save pandas dataframe
    print('video prediction finished!')


if __name__ == '__main__':
    main()
