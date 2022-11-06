from functions import *


if not os.path.exists(TRAIN_DATA) or (os.stat(TRAIN_DATA).st_size == 0):
    os.mkdir(TRAIN_DATA)
    class1 = FrameData(CLASS_1_VIDEO_DATA, TRAIN_DATA, CLASS_1_NAME)
    class2 = FrameData(CLASS_2_VIDEO_DATA, TRAIN_DATA, CLASS_2_NAME)
    class1.save_frames_from_multiple_videos()
    class2.save_frames_from_multiple_videos()
