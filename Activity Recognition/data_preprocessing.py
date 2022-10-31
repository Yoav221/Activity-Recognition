from functions import *

def main():
    
    for file in os.listdir(EXPLOSION_VIDEO_DATA):
        if file[2] == 'm':
            rename_data(EXPLOSION_VIDEO_DATA, first_num_file=1,
                        new_name='single_exlposion')
    for file in os.listdir(NON_EXPLOSION_VIDEO_DATA):
        if file[2] == 'm':
            rename_data(NON_EXPLOSION_VIDEO_DATA,
                        first_num_file=1, new_name='non_exlposion')


    if not os.path.exists(TRAIN_DATA) or (os.stat(TRAIN_DATA).st_size == 0):
        os.mkdir(TRAIN_DATA)
        SaveFrames.save_frames_from_multiple_videos(
            EXPLOSION_VIDEO_DATA, TRAIN_DATA, 'single_explosion')
        SaveFrames.save_frames_from_multiple_videos(
            NON_EXPLOSION_VIDEO_DATA, TRAIN_DATA, 'non_explosion')

if __name__ == '__main__':
    main()
    