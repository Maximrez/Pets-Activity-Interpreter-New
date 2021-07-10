import os
import shutil
from functions import process_video

if __name__ == "__main__":
    project_dir = 'D:\PycharmProjects\CV_project'
    data_dir = os.path.join(project_dir, 'data')

    test_data_dir = os.path.join(data_dir, 'test', 'videos')
    shutil.unpack_archive(os.path.join(data_dir, 'test_videos.zip'), os.path.join(data_dir, 'test'))

    file_name = "dog.mp4"
    file_path = os.path.join(test_data_dir, file_name)

    out_name = "output.avi"
    out_path = os.path.join(data_dir, out_name)

    process_video(file_path, project_dir, out_path, True, False)
