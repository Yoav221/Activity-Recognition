from functions import *
from const import *
import pandas as pd


s = SampleData(main_video_path=INPUT_VID)
frames_data, width_data, height_data = s.create_if_not_exist()
index_df = pd.DataFrame(data={'frames_data': frames_data,
                        'width_data': width_data, 'height_data': height_data})
index_df.to_csv(SAMPLE_INDEX_DF, index=False)
