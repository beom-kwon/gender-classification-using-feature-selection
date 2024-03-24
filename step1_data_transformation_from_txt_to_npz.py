import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    dataset_path_in = "./kinect gait raw dataset/"
    dataset_path_out = "./kinect gait npz dataset/"
    os.makedirs(dataset_path_out, exist_ok=True)

    label_data = pd.read_csv("person-data.csv")

    num_joints = 20  # Kinect Version 1
    person_list = os.listdir(dataset_path_in)
    for person in person_list:
        person_path_in = dataset_path_in + person + '/'
        person_path_out = dataset_path_out + person + '/'
        os.makedirs(person_path_out, exist_ok=True)

        walk_list = os.listdir(person_path_in)
        walk_list_txt = [file for file in walk_list if file.endswith(".txt")]
        for walk in walk_list_txt:
            file_path_in = person_path_in + walk
            file_path_out = person_path_out + walk[0] + ".npz"
            txt = pd.read_csv(file_path_in, sep=';', header=None).iloc[:, 1:]
            txt = np.array(txt)
            x = txt.reshape(int(txt.shape[0] / num_joints), num_joints * 3)

            if label_data["Gender"][int(person[-3:]) - 1] == 'M':
                y = 0
            elif label_data["Gender"][int(person[-3:]) - 1] == 'F':
                y = 1

            np.savez(file_path_out, x=x, y=y)