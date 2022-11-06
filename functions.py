import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from const import *
from torch.utils import data

# ------------- Prepare the data ----------------


class FrameData:

    def __init__(self, video_path, frame_path, video_name):
        self.video_path = video_path
        self.frame_path = frame_path
        self.video_name = video_name

    def save_frames_from_video_to_storage(self, video_path, frames_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                i += 1
                img_number = str(i).zfill(5)
                frame_file_name = f'{frames_path}//image{img_number}.png'
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(frame_file_name, gray_frame)
            else:
                break

    def save_frames_from_multiple_videos(self):
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)
        for i in range(len(os.listdir(self.video_path))):
            i += 1
            video_path = f'{self.video_path}/{self.sample_name}{i}.avi'
            frame_path = f'{self.frame_path}/{self.sample_name}{i}'
            if not os.path.exists(frame_path):
                os.makedirs(frame_path)
            self.save_frames_from_video_to_storage(
                video_path=video_path, frames_path=frame_path)


class SampleData(FrameData):

    def __init__(self, main_video_path=INPUT_VID, video_path=SAMPLE_VIDEO_PATH, frame_path=SAMPLE_FRAME_PATH, sample_name=SAMPLE_NAME):
        super().__init__(video_path, frame_path, sample_name)
        self.sample_name = sample_name
        self.main_video_path = main_video_path
        self.frames_data, self.width_data, self.height_data, self.df_type = None, None, None, None

    def create_samples(self, frame_jump=FRAME_JUMP, width_jump=WIDTH_JUMP,
                       height_jump=HEIGHT_JUMP, dim_limit=DIM_LIMIT):
        frames_data = []
        width_data = []
        height_data = []

        video, frame_num, frame_height, frane_width = get_video_array(
            self.main_video_path)
        vid_num = 1

        for frame_index in range(len(video) - dim_limit):
            if frame_index % frame_jump == 0:
                for width_index in range(len(video[0]) - dim_limit):
                    if width_index % width_jump == 0:
                        for height_index in range(len(video[0][0]) - dim_limit):
                            if height_index % height_jump == 0:

                                sample_video = video[frame_index:frame_index + 10, width_index: width_index + 10,
                                                     height_index:height_index+10, :]

                                frames_data.append(frame_index)
                                width_data.append(width_index)
                                height_data.append(height_index)

                                if not os.path.exists(self.video_path):
                                    os.makedirs(self.video_path)

                                if sample_video.shape == (10, 10, 10, 3):
                                    path = f'{self.video_path}/{self.sample_name}{vid_num}.avi'
                                    save_video_from_arr(sample_video, path)
                                    vid_num += 1

        return frames_data, width_data, height_data

    def create_if_not_exist(self, frame_jump=FRAME_JUMP, width_jump=WIDTH_JUMP, height_jump=HEIGHT_JUMP, dim_limit=DIM_LIMIT):

        if not os.path.exists(self.video_path) or (os.stat(self.video_path).st_size == 0):
            print("Creating Sample Videos ----->")
            self.frames_data, self.width_data, self.height_data = self.create_samples(
                frame_jump=frame_jump, width_jump=width_jump, height_jump=height_jump, dim_limit=dim_limit)
            print('Samples Videos have been created!')
            self.df_type = 1
        else:
            print('Sample videos already exist!')

        if not os.path.exists(self.frame_path) or (os.stat(self.frame_path).st_size == 0):
            print("Creating Sample Frames ----->")
            self.save_frames_from_multiple_videos()
            print('Samples Frames have been created!')
        else:
            print('Sample frames already exist!')
            self.df_type = 0

        return self.frames_data, self.width_data, self.height_data, self.df_type


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


def rename_data(data, first_num_file, new_name):
    samples = []
    for file in os.listdir(data):
        if file[0] == 's':
            samples.append(file)

    for x in range(len(samples)):
        os.rename(f'{data}/{samples[x]}',
                  f'{data}/{new_name}{x + first_num_file}.avi')


def create_X_list_and_actions(data_path, loc1='v_', loc2='_g', ):
    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find(loc1)
        loc2 = f.find(loc2)
        actions.append(f[(loc1 + 2): loc2])
        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    return all_X_list, actions, fnames


def convert2label_and_hot(action_names):
    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)
    list(le.classes_)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    return le, enc


def data_loading_params(batch_size):
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    params = {'batch_size': batch_size, 'shuffle': False,
              'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    return use_cuda, device, params


def get_video_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = np.empty((frame_count, frame_height,
                     frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, video[fc] = cap.read()
        fc += 1
    cap.release()
    return video, frame_count, frame_height, frame_width


def save_video_from_arr(vid_arr, path):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(
        *'MJPG'), 25, (vid_arr.shape[2], vid_arr.shape[1]))
    for i in range(vid_arr.shape[0]):
        data = vid_arr[i, :, :, :]
        out.write(data)
    out.release()


# --------------- Train & Validation --------------------
def train(log_interval, model, device, train_loader, optimizer, epoch):
    # Set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        # output has dim = (batch, number of classes)
        output = rnn_decoder(cnn_encoder(X))

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score([y.cpu().data.squeeze().numpy()], [
                                    y_pred.cpu().data.squeeze().numpy()])
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()
        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader, epoch):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            # (y_pred != output) get the index of the max log-probability
            y_pred = output.max(1, keepdim=True)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze(
    ).numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        len(all_y), test_loss, 100 * test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path,
               'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path,
               'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path,
               'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score
