from functions import *

# Make the videos ready for training - creating frame seq out of the videos
if not os.path.exists(TRAIN_DATA) or (os.stat(TRAIN_DATA).st_size == 0):
    os.mkdir(TRAIN_DATA)
    save_frames_from_multiple_videos(CLASS_1_VIDEO_DATA, TRAIN_DATA, CLASS_1_NAME)
    save_frames_from_multiple_videos(CLASS_2_VIDEO_DATA, TRAIN_DATA, CLASS_2_NAME)
