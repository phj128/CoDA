import torch
import numpy as np
import coda.utils.matrix as matrix


def humanoid_params_to_localmat(humanoid_params):
    pose = humanoid_params["pose"]  # (N, 52, 3)
    local_pos = humanoid_params["local_pos"]  # (52, 3)
    pose_rotmat = matrix.axis_angle_to_matrix(pose)  # (N, 52, 3, 3)
    local_mat = matrix.get_TRS(pose_rotmat, local_pos)
    return local_mat


def load_arctic_data(path):
    data = torch.load(path, map_location="cpu")
    return data


def get_humanoid_data(data):
    local_mat = humanoid_params_to_localmat(data["humanoid"])
    global_mat = matrix.forward_kinematics(local_mat, data["humanoid"]["parent"])  # (N, J, 4, 4)
    return local_mat, global_mat


def get_obj_data(data):
    local_mat = humanoid_params_to_localmat(data["obj"])
    global_mat = matrix.forward_kinematics(local_mat, data["obj"]["parent"])  # (N, J, 4, 4)
    return local_mat, global_mat


def aztoay(mat):
    trans_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trans_matrix = torch.tensor(trans_matrix, dtype=mat.dtype, device=mat.device)  # (1, 4, 4)
    mat = matrix.get_mat_BfromA(trans_matrix[None], mat)
    root0 = matrix.get_position(mat[0, 0])  # (3,)
    new_pos = matrix.get_position(mat).clone()  # (N, J, 3)
    new_pos[..., 0] -= root0[0]
    new_pos[..., 2] -= root0[2]
    mat = matrix.set_position(mat, new_pos)

    return mat


def aytoaz(verts):
    trans_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trans_matrix = torch.tensor(trans_matrix, dtype=verts.dtype, device=verts.device)  # (1, 4, 4)
    new_verts = matrix.get_position_from(verts, trans_matrix)
    return new_verts


def aztoay_hoi(humanoid_mat, obj_mat):
    # az to ay
    trans_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    trans_matrix = torch.tensor(trans_matrix, dtype=humanoid_mat.dtype, device=humanoid_mat.device)  # (1, 4, 4)
    humanoid_mat = matrix.get_mat_BfromA(trans_matrix[None], humanoid_mat)

    # ay to ayfz
    humanoid_rotmat = matrix.get_rotation(humanoid_mat)  # (N, 3, 3)
    root_quat = matrix.quat_wxyz2xyzw(matrix.matrix_to_quaternion(humanoid_rotmat[..., 0, :, :]))  # (N, 4)
    ay_root_quat = matrix.calc_heading_quat(root_quat, head_ind=2, gravity_axis="y")  # (N, 4)
    ay_root_rotmat = matrix.quaternion_to_matrix(matrix.quat_xyzw2wxyz(ay_root_quat))  # (N, 3, 3)
    root_pos = matrix.get_position(humanoid_mat[:1, 0, :, :])  # (1, 3)
    root_zeropos_mat = matrix.get_TRS(ay_root_rotmat[:1], torch.zeros_like(root_pos))  # (1, 4, 4)
    humanoid_mat = matrix.get_mat_BtoA(root_zeropos_mat, humanoid_mat)

    root0 = matrix.get_position(humanoid_mat[0, 0]).clone()  # (3,)

    new_pos = matrix.get_position(humanoid_mat).clone()  # (N, J, 3)
    new_pos[..., 0] -= root0[0]
    new_pos[..., 2] -= root0[2]
    humanoid_mat = matrix.set_position(humanoid_mat, new_pos)

    obj_mat = matrix.get_mat_BfromA(trans_matrix[None], obj_mat)
    obj_mat = matrix.get_mat_BtoA(root_zeropos_mat, obj_mat)
    new_pos = matrix.get_position(obj_mat).clone()  # (N, J, 3)
    new_pos[..., 0] -= root0[0]
    new_pos[..., 2] -= root0[2]
    obj_mat = matrix.set_position(obj_mat, new_pos)

    ### add z-axis translation to obj
    # delta_pos = torch.zeros_like(new_pos) # (N, J, 3)
    # delta_pos[..., 2] = 0.3 / 30
    # delta_pos = torch.cumsum(delta_pos, dim=0)
    # obj_mat = matrix.set_position(obj_mat, new_pos + delta_pos)
    ############################################################
    return humanoid_mat, obj_mat


# for smplx humanoid
def get_wrist_trajectory(global_mat):
    wrist_inds = [17, 41]  # left right
    wrist_trajectory = matrix.get_position(global_mat[..., wrist_inds, :, :])  # (N, 2, 3)
    return wrist_trajectory


def get_finger_trajectory(global_mat, is_right=False, istip=False):
    # index, middle, pinky, ring, thumb
    if istip:
        if is_right:
            figner_inds = [45, 49, 53, 57, 61]
        else:
            figner_inds = [21, 25, 29, 33, 37]
    else:
        if is_right:
            figner_inds = [44, 48, 52, 56, 60]
        else:
            figner_inds = [20, 24, 28, 32, 36]
    finger_trajectory = matrix.get_position(global_mat[..., figner_inds, :, :])  # (N, 5, 3)
    return finger_trajectory


def get_handpose(mat, is_right=False, istip=False):
    if istip:
        if is_right:
            finger_inds = [
                42,
                43,
                44,
                45,  # index
                46,
                47,
                48,
                49,  # middle
                50,
                51,
                52,
                53,  # pinky
                54,
                55,
                56,
                57,  # ring
                58,
                59,
                60,
                61,  # thumb
            ]
        else:
            finger_inds = [
                18,
                19,
                20,
                21,  # index
                22,
                23,
                24,
                25,  # middle
                26,
                27,
                28,
                29,  # pinky
                30,
                31,
                32,
                33,  # ring
                34,
                35,
                36,
                37,  # thumb
            ]
    else:
        if is_right:
            finger_inds = [
                #    41, # wrist
                42,
                43,
                44,  # index
                46,
                47,
                48,  # middle
                50,
                51,
                52,  # pinky
                54,
                55,
                56,  # ring
                58,
                59,
                60,  # thumb
            ]
        else:
            finger_inds = [
                #    17, # wrist
                18,
                19,
                20,  # index
                22,
                23,
                24,  # middle
                26,
                27,
                28,  # pinky
                30,
                31,
                32,  # ring
                34,
                35,
                36,  # thumb
            ]
    return mat[..., finger_inds, :, :]  # (N, 16, 3, 3)


def get_handposition(pos, is_right=False, istip=False):
    if istip:
        if is_right:
            finger_inds = [
                42,
                43,
                44,
                45,  # index
                46,
                47,
                48,
                49,  # middle
                50,
                51,
                52,
                53,  # pinky
                54,
                55,
                56,
                57,  # ring
                58,
                59,
                60,
                61,  # thumb
            ]
        else:
            finger_inds = [
                18,
                19,
                20,
                21,  # index
                22,
                23,
                24,
                25,  # middle
                26,
                27,
                28,
                29,  # pinky
                30,
                31,
                32,
                33,  # ring
                34,
                35,
                36,
                37,  # thumb
            ]
    else:
        if is_right:
            finger_inds = [
                #    41, # wrist
                42,
                43,
                44,  # index
                46,
                47,
                48,  # middle
                50,
                51,
                52,  # pinky
                54,
                55,
                56,  # ring
                58,
                59,
                60,  # thumb
            ]
        else:
            finger_inds = [
                #    17, # wrist
                18,
                19,
                20,  # index
                22,
                23,
                24,  # middle
                26,
                27,
                28,  # pinky
                30,
                31,
                32,  # ring
                34,
                35,
                36,  # thumb
            ]
    return pos[..., finger_inds, :]  # (N, 16, 3)


####################

# SMPLX HUMANOID RELATED #
ALL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

# SMPLX ORDER
SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    # "jaw",
    # "left_eye_smplhf",
    # "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    # "left_index",
    # "left_middle",
    # "left_pinky",
    # "left_ring",
    # "left_thumb",
    # "right_index",
    # "right_middle",
    # "right_pinky",
    # "right_ring",
    # "right_thumb",
]


SMPLXTIP_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    # "jaw",
    # "left_eye_smplhf",
    # "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "left_index",
    "left_middle",
    "left_pinky",
    "left_ring",
    "left_thumb",
    "right_index",
    "right_middle",
    "right_pinky",
    "right_ring",
    "right_thumb",
]

# SMPLX in HUMANOID order
SMPLXHUMANOID_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_index",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_middle",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_pinky",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_ring",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_index",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_middle",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_pinky",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_ring",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb",
]

HUMANOID2SMPLX = [SMPLXHUMANOID_JOINT_NAMES.index(name) for name in SMPLX_JOINT_NAMES]
SMPLXTIP2HUMANOID = [SMPLXTIP_JOINT_NAMES.index(name) for name in SMPLXHUMANOID_JOINT_NAMES]

# SMPLH PARENTS
PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
]
