from functions import *
    

for file in os.listdir(CLASS_1_VIDEO_DATA):
    if file[:len(SAMPLE_NAME)] == SAMPLE_NAME:
        rename_data(CLASS_1_VIDEO_DATA, first_num_file=1, new_name=CLASS_1_NAME)

for file in os.listdir(CLASS_2_VIDEO_DATA):
    if file[:len(SAMPLE_NAME)] == SAMPLE_NAME:
        rename_data(CLASS_2_VIDEO_DATA, first_num_file=1, new_name=CLASS_2_NAME)

# Make the videos ready for training - creating frame seq out of the videos
if not os.path.exists(TRAIN_DATA) or (os.stat(TRAIN_DATA).st_size == 0):
    os.mkdir(TRAIN_DATA)
    save_frames_from_multiple_videos(CLASS_1_VIDEO_DATA, TRAIN_DATA, CLASS_1_NAME)
    save_frames_from_multiple_videos(CLASS_2_VIDEO_DATA, TRAIN_DATA, CLASS_2_NAME)


    
