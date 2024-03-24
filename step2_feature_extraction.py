import numpy as np
import scipy
import os


def SkeletonInfo():
    num_joints = 20
    limb_info = [[2, 3],  # 01. shoulder right
                 [3, 5],  # 02. arm right
                 [5, 7],  # 03. forearm right
                 [7, 9],  # 04. hand right
                 [12, 13],  # 05. hip right
                 [13, 15],  # 06. thigh right
                 [15, 17],  # 07. leg right
                 [17, 19],  # 08. foot right
                 [1, 2],  # 09. neck
                 [2, 11],  # 10. upper spine
                 [11, 12],  # 11. lower spine
                 [2, 4],  # 12. shoulder left
                 [4, 6],  # 13. arm left
                 [6, 8],  # 14. forearm left
                 [8, 10],  # 15. hand left
                 [12, 14],  # 16. hip left
                 [14, 16],  # 17. thigh left
                 [16, 18],  # 18. leg left
                 [18, 20]]  # 19. foot left

    return num_joints, np.array(limb_info)


def feature_extraction_F1(motion, num_joints, limb_info):
    # Title: Gender and body mass index classification using a Microsoft Kinect sensor
    # Author: Andersson et al.
    # Year: 2015
    # Kinect Version 1
    # 1) Average length of Each Body Part -> Dim. = 19
    # 2) Average Height -> Dim. = 1

    feature = []
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)

        vec = []
        # Calculate the Length of Each Body Part
        for i in range(0, limb_info.shape[0], 1):
            vec.append(np.linalg.norm(pose[limb_info[i, 0] - 1, :] - pose[limb_info[i, 1] - 1, :]))

        # Calculate the Height of Subject
        # height = neck + upper spine + lower spine
        #        + avg(right hip, left hip)
        #        + avg(right thigh, left thigh)
        #        + avg(right leg, left leg)
        neck = vec[9 - 1]  # neck
        upper_spine = vec[10 - 1]  # upper_spine
        lower_spine = vec[11 - 1]  # lower_spine
        avg_hips = (vec[5 - 1] + vec[16 - 1]) / 2  # average of hips
        avg_thighs = (vec[6 - 1] + vec[17 - 1]) / 2  # average of thighs
        avg_legs = (vec[7 - 1] + vec[18 - 1]) / 2  # average of legs
        subj_height = neck + upper_spine + lower_spine + avg_hips + avg_thighs + avg_legs
        vec.append(subj_height)

        feature.append(vec)

    feature = np.array(feature)
    return np.mean(feature, axis=0)


def feature_extraction_F2(motion, num_joints):
    # Title: Gait with a combination of swing arm feature extraction for gender identification using Kinect skeleton
    # Author: Bachtiar et al.
    # Year: 2016
    # Kinect Version 1
    # 1) Average Width between Right and Left Feet -> Dim. = 1
    # 2) Average Distance between Right and Left Hands -> Dim. = 1

    feature = []
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)

        vec = []
        # Calculate the Width between the Right and Left Feet
        vec.append(np.linalg.norm(pose[19 - 1, :] - pose[20 - 1, :]))

        # Calculate the Distance between the Right and Left Hands
        vec.append(np.linalg.norm(pose[9 - 1, :] - pose[10 - 1, :]))

        feature.append(vec)

    feature = np.array(feature)
    return np.mean(feature, axis=0)


def feature_extraction_F3(motion, num_joints):
    # Title: Human gender classification based on gait features using kinect sensor
    # Author: Ahmed and Sabir
    # Year: 2017
    # Kinect Version 1
    # 1) Distance between the Right and Left Shoulders -> Avg., Std., Skewness -> Dim. = 3
    # 2) Distance between the Right and Left Ankles -> Avg., Std., Skewness -> Dim. = 3

    feature = []
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)

        vec = []
        # Calculate the Distance between the Right and Left Shoulders
        vec.append(np.linalg.norm(pose[3 - 1, :] - pose[4 - 1, :]))

        # Calculate the Distance between the Right and Left Ankles
        vec.append(np.linalg.norm(pose[17 - 1, :] - pose[18 - 1, :]))

        feature.append(vec)

    feature = np.array(feature)

    return np.concatenate((np.mean(feature, axis=0), np.std(feature, axis=0), scipy.stats.skew(feature, axis=0)))


def feature_extraction_F4(motion, num_joints):
    # Title: Gender detection using 3D anthropometric measurements by Kinect
    # Author: Camalan et al.
    # Year: 2018
    # Kinect Version 1
    # 1) Average HbS -> Dim. = 1
    # 2) Average HbE -> Dim. = 1
    # 3) Average HbW -> Dim. = 1

    feature = []
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)

        vec = []
        # Calculate the Height by Skeleton(HbS)
        # HbS = L_H_CS + L_CS_S + L_S_CH + L_CH_K + L_K_A + L_A_F
        L_H_CS = np.linalg.norm(pose[1 - 1, :] - pose[2 - 1, :])
        L_CS_S = np.linalg.norm(pose[2 - 1, :] - pose[11 - 1, :])
        L_S_CH = np.linalg.norm(pose[11 - 1, :] - pose[12 - 1, :])
        L_CH_K = (np.linalg.norm(pose[12 - 1, :] - pose[15 - 1, :]) + np.linalg.norm(
            pose[12 - 1, :] - pose[16 - 1, :])) / 2
        L_K_A = (np.linalg.norm(pose[15 - 1, :] - pose[17 - 1, :]) + np.linalg.norm(
            pose[16 - 1, :] - pose[18 - 1, :])) / 2
        L_A_F = (np.linalg.norm(pose[17 - 1, :] - pose[19 - 1, :]) + np.linalg.norm(
            pose[18 - 1, :] - pose[20 - 1, :])) / 2

        HbS = L_H_CS + L_CS_S + L_S_CH + L_CH_K + L_K_A + L_A_F
        vec.append(HbS)

        # Calculate the Height by Estimation(HbE)
        # the distance between shoulders and knees = 0.52 × height
        # HbE = the distance between shoulders and knees / 0.52
        HbE = (np.linalg.norm(pose[3 - 1, :] - pose[15 - 1, :]) + np.linalg.norm(
            pose[4 - 1, :] - pose[16 - 1, :])) / 2 / 0.52
        vec.append(HbE)

        # Calculate the Height by Wingspan(HbW)
        # HbW = L_LH_LW + L_LW_LE + L_LE_LS + L_LS_CS
        #     + L_CS_RS + L_RS_RE + L_RE_RW + L_RW_RH
        L_LH_LW = np.linalg.norm(pose[10 - 1, :] - pose[8 - 1, :])
        L_LW_LE = np.linalg.norm(pose[8 - 1, :] - pose[6 - 1, :])
        L_LE_LS = np.linalg.norm(pose[6 - 1, :] - pose[4 - 1, :])
        L_LS_CS = np.linalg.norm(pose[4 - 1, :] - pose[2 - 1, :])
        L_CS_RS = np.linalg.norm(pose[2 - 1, :] - pose[3 - 1, :])
        L_RS_RE = np.linalg.norm(pose[3 - 1, :] - pose[5 - 1, :])
        L_RE_RW = np.linalg.norm(pose[5 - 1, :] - pose[7 - 1, :])
        L_RW_RH = np.linalg.norm(pose[7 - 1, :] - pose[9 - 1, :])

        HbW = L_LH_LW + L_LW_LE + L_LE_LS + L_LS_CS + L_CS_RS + L_RS_RE + L_RE_RW + L_RW_RH
        vec.append(HbW)

        feature.append(vec)

    feature = np.array(feature)
    return np.mean(feature, axis=0)


def calc_dist_T(pose, idx):
    idx = idx - 1
    return np.abs(pose[12 - 1, 2] - pose[idx, 2])


def calc_dist_F(pose, idx):
    idx = idx - 1
    numerator = np.abs((pose[14 - 1, 1] - pose[13 - 1, 1]) * (pose[idx, 1] - pose[12 - 1, 0]) +
                       (pose[13 - 1, 0] - pose[14 - 1, 0]) * (pose[idx, 2] - pose[12 - 1, 1]))
    denominator = np.sqrt((pose[14 - 1, 1] - pose[13 - 1, 1]) ** 2 + (pose[13 - 1, 0] - pose[14 - 1, 0]) ** 2)
    return numerator / denominator


def calc_dist_M(pose, idx):
    idx = idx - 1
    numerator = np.abs((pose[14 - 1, 0] - pose[13 - 1, 0]) * (pose[idx, 0] - pose[12 - 1, 0]) +
                       (pose[14 - 1, 1] - pose[13 - 1, 1]) * (pose[idx, 1] - pose[12 - 1, 1]) +
                       (pose[14 - 1, 2] - pose[13 - 1, 2]) * (pose[idx, 2] - pose[12 - 1, 2]))
    denominator = np.sqrt((pose[14, 0] - pose[13, 0]) ** 2 +
                          (pose[14, 1] - pose[13, 1]) ** 2 +
                          (pose[14, 2] - pose[13, 1]) ** 2)
    return numerator / denominator


def feature_extraction_F5(motion, num_joints):
    # Title: Joint Swing Energy for Skeleton-Based Gender Classification
    # Author: Kwon and Lee
    # Year: 2021
    # Kinect Version 1
    # Joint Swing Energy(JSE) -> Dim. = 51
    D_T = np.zeros(shape=(motion.shape[0], 17))  # Transverse Plane
    D_F = np.zeros(shape=(motion.shape[0], 17))  # Frontal Plane
    D_M = np.zeros(shape=(motion.shape[0], 17))  # Median Plane
    for frm in range(0, motion.shape[0], 1):
        pose = motion[frm, :].reshape(num_joints, 3)  # Current Pose
        pose = np.column_stack((pose[:, 2], pose[:, 0], pose[:, 1]))
        # Target Joints
        # 1	Head
        # 2	Shoulder-Center
        # 3	Shoulder-Right
        # 4	Shoulder-Left
        # 5	Elbow-Right
        # 6	Elbow-Left
        # 7	Wrist-Right
        # 8	Wrist-Left
        # 9	Hand-Right
        # 10	Hand-Left
        # 11	Spine
        # 12	Hip-centro <--- Not a Target
        # 13	Hip-Right  <--- Not a Target
        # 14	Hip-Left   <--- Not a Target
        # 15	Knee-Right
        # 16	Knee-Left
        # 17	Ankle-Right
        # 18	Ankle-Left
        # 19	Foot-Right
        # 20	Foot-Left

        J = np.concatenate((np.arange(1, 12, 1), np.arange(15, 21, 1)))
        cnt = 0
        for j_idx in J:
            D_T[frm, cnt] = calc_dist_T(pose, j_idx);
            D_F[frm, cnt] = calc_dist_F(pose, j_idx);
            D_M[frm, cnt] = calc_dist_M(pose, j_idx);
            cnt = cnt + 1

    # NaN Check
    frm_D_T = []
    frm_D_F = []
    frm_D_M = []
    for frm in range(0, motion.shape[0], 1):
        if True in np.isnan(D_T[frm, :]):
            frm_D_T.append(frm)
        if True in np.isnan(D_F[frm, :]):
            frm_D_F.append(frm)
        if True in np.isnan(D_M[frm, :]):
            frm_D_M.append(frm)

    D_T = np.delete(D_T, obj=frm_D_T, axis=0)
    D_F = np.delete(D_F, obj=frm_D_F, axis=0)
    D_M = np.delete(D_M, obj=frm_D_M, axis=0)

    R_T = np.mean(D_T, axis=0)
    R_F = np.mean(D_F, axis=0)
    R_M = np.mean(D_M, axis=0)

    JSE = np.concatenate((R_T, R_F, R_M))
    return JSE


if __name__ == "__main__":
    num_joints, limb_info = SkeletonInfo()

    dataset_path_in = "./kinect gait npz dataset/"

    person_id, x, y = [], [], []
    person_list = os.listdir(dataset_path_in)
    for person in person_list:
        person_path_in = dataset_path_in + person + '/'

        walk_list = os.listdir(person_path_in)
        walk_list_txt = [file for file in walk_list if file.endswith(".npz")]
        for walk in walk_list_txt:
            file_path_in = person_path_in + walk
            data = np.load(file_path_in)

            motion = data['x']
            label = data['y']

            # Feature Extraction
            F1 = feature_extraction_F1(motion, num_joints, limb_info)  # Dim. = 20
            F2 = feature_extraction_F2(motion, num_joints)  # Dim. = 2
            F3 = feature_extraction_F3(motion, num_joints)  # Dim. = 6
            F4 = feature_extraction_F4(motion, num_joints)  # Dim. = 3
            F5 = feature_extraction_F5(motion, num_joints)  # Dim. = 51

            person_id.append(int(person[-3:]))
            x.append(np.concatenate((F1, F2, F3, F4, F5)))
            y.append(label)

    np.savez("./feature.npz", person_id=np.array(person_id), x=np.array(x), y=np.array(y))
    print(np.array(x).shape, np.array(y).shape)