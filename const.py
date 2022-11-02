# The path the the model is saved on
from torch import dropout


save_model_path = r'CRNN_ckpt'

# Train data - this is the data we use to train the model
TRAIN_DATA = r'./Data/Train_data'

# Video data - you can change to your classes
EXPLOSION_VIDEO_DATA = r'./Data/Video_data/class1'
NON_EXPLOSION_VIDEO_DATA = r'./Data/Video_data/class2'

# Samples Data - the samples that have been splited from the whole video, and we run our model on those samples
SAMPLE_VIDEO_PATH = '...'
SAMPLE_FRAME_PATH = '...'
SAMPLE_INDEX_DF = '...'
# INPUT VIDEO
INPUT_VID = r'...'

# Names
CLASSES = ['single_explosion', 'non_explosion']
EXPLOSION = 'single_explosion'
NON_EXPLOSION = 'non_explosion'

# Create samples
DIM_LIMIT = 10
FRAME_JUMP = 10
WIDTH_JUMP = 10
HEIGHT_JUMP = 10
SAMPLE_NAME = 'sample_'

# EndcoderCNN hyperparams
CNN_embed_dim = 48
img_x, img_y = 10, 10
dropout_p = 0.2

# DecoderRNN hyperparams
RNN_hidden_layers = 1
RNN_hidden_nodes = 64
FC_dim = 24
RNN_FC_dim = 36

# Training params
k = 2
EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
LOG_INTERVAL = 10
begin_frame, end_frame, skip_frame = 1, 11, 1
