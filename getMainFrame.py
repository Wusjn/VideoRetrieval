import os
import cv2
import numpy as np

def get_all_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_histogram(frame):
    f = np.array(frame).transpose()
    histogram = [[0] * 256, [0] *256, [0]* 256]
    for i, rgb_f in enumerate(f):
        for line in rgb_f:
            for pixel in line:
                histogram[i][pixel] += 1
    return histogram


def get_shot_first_frame_index(all_frames):
    max_shot_num = 3
    all_histograms = []
    for frame in all_frames:
        all_histograms.append(get_histogram(frame))

    diff_list = []
    for i in range(len(all_histograms) - 1):
        prev = np.array(all_histograms[i])
        next = np.array(all_histograms[i + 1])
        diff = np.abs(prev - next).sum()
        diff_list.append(diff)
    diff_list = np.array(diff_list)

    median = np.median(diff_list)
    mean = np.mean(diff_list)
    shots_first_frame_index = []
    for i, diff in enumerate(diff_list):
        if diff > 3 * median and diff > 3 * mean:
            shots_first_frame_index.append((diff, i+1))

    if len(shots_first_frame_index) + 1 > max_shot_num:
        sorted(shots_first_frame_index, key=lambda x:x[0], reverse=True)
        shots_first_frame_index = shots_first_frame_index[:max_shot_num - 1]
        shots_first_frame_index = list(map(lambda x:x[1], shots_first_frame_index))
        sorted(shots_first_frame_index)
    else:
        shots_first_frame_index = list(map(lambda x: x[1], shots_first_frame_index))

    shots_first_frame_index = [0] + shots_first_frame_index + [len(all_frames)]
    return shots_first_frame_index

def save_main_frames(video_path, output_dir):
    all_frames = get_all_frames(video_path)
    shots_first_frame_index = get_shot_first_frame_index(all_frames)


    for i in range(len(shots_first_frame_index) - 1):
        head_index = shots_first_frame_index[i]
        tail_index = shots_first_frame_index[i+1] - 1
        medium_index = int((head_index + tail_index) / 2)
        one_quarter = int((head_index + medium_index) / 2)
        three_quarter = int((tail_index + medium_index) / 2)
        shot_lenth = tail_index - head_index + 1

        cv2.imwrite(output_dir + "/shot_" + str(i) + "_len_" + str(shot_lenth) + "_head.jpg", all_frames[one_quarter])
        cv2.imwrite(output_dir + "/shot_" + str(i) + "_len_" + str(shot_lenth) + "_medium.jpg", all_frames[medium_index])
        cv2.imwrite(output_dir + "/shot_" + str(i) + "_len_" + str(shot_lenth) + "_tail.jpg", all_frames[three_quarter])




def mainFramesExtraction(HMDB_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for type_name in os.listdir(HMDB_dir):

        type_dir = HMDB_dir + "/" + type_name
        output_type_dir = output_dir + "/" + type_name
        if not os.path.exists(output_type_dir):
            os.mkdir(output_type_dir)

        for video_name in os.listdir(type_dir):

            video_path = type_dir + "/" + video_name
            output_video_dir = output_type_dir + "/" + video_name
            if not os.path.exists(output_video_dir):
                os.mkdir(output_video_dir)

            save_main_frames(video_path, output_video_dir)
            print(output_video_dir)

if __name__ == '__main__':
    mainFramesExtraction("./data/HMDB", "./data/HMDB_main_frame")
    #save_main_frames("./data/HMDB/jump/Learn_Freerunning_and_Parkour_-_Diving_Roll_jump_f_cm_np1_le_bad_4.avi","./")
    #save_main_frames("./data/HMDB/brush_hair/Brushing_hair__the_right_way_brush_hair_u_nm_np1_fr_goo_2.avi","./")
    #save_main_frames("./data/HMDB/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi", "./")