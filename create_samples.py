from re import X
from functions import *
from const import *
import pandas as pd


def create_if_not_exist(input_vid=INPUT_VID, sample_video_path=SAMPLE_VIDEO_PATH, sample_frame_path=SAMPLE_FRAME_PATH, sample_name=SAMPLE_NAME):
    if not os.path.exists(sample_video_path) or (os.stat(sample_video_path).st_size == 0):
        print("Creating Sample Videos ----->")
        ()

        frames_data, width_data, height_data = create_samples(main_video_path=input_vid, explosion_name='single_explosion',
                                                              non_explosion_name='non_explosion', sample_name=sample_name,
                                                              frame_jump=FRAME_JUMP, width_jump=WIDTH_JUMP, height_jump=HEIGHT_JUMP, dim_limit=DIM_LIMIT)
        print('Samples Videos have been created!')

    else:
        print('Sample videos already exist!')

    if not os.path.exists(sample_frame_path) or (os.stat(sample_frame_path).st_size == 0):
        print("Creating Sample Frames ----->")
        save_frames_from_multiple_videos(
            sample_name, sample_video_path, sample_frame_path)
        print('Samples Frames have been created!')

    else:
        print('Sample frames already exist!')


x, y, z = create_if_not_exist()
index_df = pd.DataFrame(
    data={'frame_index': x, 'width_index': y, 'hegith_index': z})
index_df.to_csv(SAMPLE_INDEX_DF, index=False)
