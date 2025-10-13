import torch
from tqdm import tqdm
from pathlib import Path
from coda.utils.bps import bps_gen_ball_inside, calculate_bps, get_pc_center, get_pc_scale
from coda.utils.arctic.object_tensors import ObjectTensors
from coda.dataset.arctic.utils import *


def process_bps(humanoid_globalmat, obj_globalmat, angles, obj_name, is_mirror=False):
    left_finger_traj = get_finger_trajectory(humanoid_globalmat, is_right=False, istip=IS_TIP)  # (N, 5, 3)
    right_finger_traj = get_finger_trajectory(humanoid_globalmat, is_right=True, istip=IS_TIP)  # (N, 5, 3)
    wrist_traj = get_wrist_trajectory(humanoid_globalmat)  # (N, 2, 3)
    ori_finger_traj = torch.cat([wrist_traj, left_finger_traj, right_finger_traj], dim=-2)  # (N, 12, 3)
    if is_mirror:
        wrist_traj = matrix.mirror_pos_yzplane(wrist_traj)
        wrist_traj = torch.cat([wrist_traj[..., [1], :], wrist_traj[..., [0], :]], dim=-2)
        right_finger_traj_ = matrix.mirror_pos_yzplane(left_finger_traj)
        left_finger_traj = matrix.mirror_pos_yzplane(right_finger_traj)
        right_finger_traj = right_finger_traj_
    finger_traj = torch.cat([wrist_traj, left_finger_traj, right_finger_traj], dim=-2)  # (N, 12, 3)

    top_mask, bottom_mask = object_tensor.get_part_obj_mask(obj_name)

    arctic_obj_data = {}
    arctic_obj_data["angles"] = torch.zeros_like(angles[:1])
    arctic_obj_data["transl"] = torch.zeros_like(matrix.get_position(obj_globalmat[:1, 0, :, :]))
    arctic_obj_data["global_orient"] = torch.zeros_like(
        matrix.matrix_to_axis_angle(matrix.get_rotation(obj_globalmat[:1, 0, :, :]))
    )
    with torch.no_grad():
        obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
        obj_out = object_tensor(**arctic_obj_data, query_names=obj_ns)
    obj_verts = obj_out["v"]
    obj_faces = obj_out["f"][0]
    if is_mirror:
        obj_verts = matrix.mirror_pos_yzplane(obj_verts)

    center = get_pc_center(obj_verts[0])  # (3, )
    scale = get_pc_scale(obj_verts[0], multiply=1.25)  # (1, )
    obj_verts = obj_verts - center
    top_verts = obj_verts[..., top_mask, :]  # (1, M, 3)
    bottom_verts = obj_verts[..., bottom_mask, :]  # (1, M, 3)
    scaled_obj_local_top_verts = top_verts / scale
    scaled_obj_local_bottom_verts = bottom_verts / scale
    idle_bps_bottom, bps_bottom_ind = calculate_bps(basis_point[None], scaled_obj_local_bottom_verts, return_xyz=False)
    idle_bps_top, bps_top_ind = calculate_bps(basis_point[None], scaled_obj_local_top_verts, return_xyz=False)

    arctic_obj_data = {}
    arctic_obj_data["angles"] = angles
    arctic_obj_data["transl"] = matrix.get_position(obj_globalmat[..., 0, :, :])
    arctic_obj_data["global_orient"] = matrix.matrix_to_axis_angle(matrix.get_rotation(obj_globalmat[..., 0, :, :]))
    with torch.no_grad():
        obj_ns = [obj_name for _ in range(arctic_obj_data["angles"].shape[0])]
        obj_out = object_tensor(**arctic_obj_data, query_names=obj_ns)
    obj_verts = obj_out["v"]
    obj_verts = obj_verts.cuda()  # (N, M, 3)
    if is_mirror:
        ori_verts = obj_verts.clone()
        obj_verts = matrix.mirror_pos_yzplane(obj_verts)
    # moved_center = get_pc_center(obj_verts)
    obj_faces = obj_out["f"][0]

    if is_mirror:
        ori_globalmat = obj_globalmat.clone()
        obj_globalmat = matrix.mirror_mat_yzplane(obj_globalmat)

    global_obj_center = matrix.get_position_from(center[None], obj_globalmat[..., 0, :, :])  # (N, 3)
    global_obj_center_mat = matrix.get_TRS(
        matrix.get_rotation(obj_globalmat[..., 0, :, :]), global_obj_center
    )  # (N, 4, 4)
    top_verts = obj_verts[..., top_mask, :]
    bottom_verts = obj_verts[..., bottom_mask, :]
    obj_local_top_verts = matrix.get_relative_position_to(top_verts, global_obj_center_mat)  # (N, M, 3)
    obj_local_bottom_verts = matrix.get_relative_position_to(bottom_verts, global_obj_center_mat)  # (N, M, 3)
    scaled_obj_local_top_verts = obj_local_top_verts / scale
    scaled_obj_local_bottom_verts = obj_local_bottom_verts / scale
    bps_bottom, bps_bottom_ind = calculate_bps(basis_point[None], scaled_obj_local_bottom_verts, return_xyz=False)
    bps_top, bps_top_ind = calculate_bps(basis_point[None], scaled_obj_local_top_verts, return_xyz=False)

    obj_local_finger_traj = matrix.get_relative_position_to(finger_traj, global_obj_center_mat)  # (N, 10, 3)
    scaled_obj_local_finger_traj = obj_local_finger_traj / scale  # (N, 10, 3)

    finger_delta_pos = basis_point_finger[None, :, None] - scaled_obj_local_finger_traj[:, None]  # (N, M, 10, 3)
    finger_delta_dist = torch.norm(finger_delta_pos, dim=-1)  # (N, M, 10)

    finger_verts_dist = obj_verts[..., :, None, :] - finger_traj[..., None, 2:, :]  # (N, M, 10, 3)
    finger_verts_dist = torch.norm(finger_verts_dist, dim=-1)  # (N, M, 10)
    close_mask = finger_verts_dist.min(dim=-2)[0] < 0.05  # (N, 10)
    finger_verts_dist = finger_verts_dist.min(dim=-2)[0]  # (N, 10)

    data_dict = {
        "finger_dist": finger_delta_dist.cpu(),
        "finger_vert_dist": finger_verts_dist.cpu(),
        "close_mask": close_mask.cpu(),
        "bottom_bps": bps_bottom.cpu(),
        "top_bps": bps_top.cpu(),
        "scale": scale.cpu(),
        "center": center.cpu(),
        "idle_bottom_bps": idle_bps_bottom.cpu(),
        "idle_top_bps": idle_bps_top.cpu(),
    }
    return data_dict


N_POINT = 256
basis_point = bps_gen_ball_inside(n_bps=N_POINT, random_seed=100)  # (M, 3)
basis_point = basis_point.cuda()

basis_point_finger = basis_point.clone()

object_tensor = ObjectTensors().cuda()

dataset_path = "./inputs/arctic_neutral"

save_bps_data = {}
save_bps_data["basis_point"] = basis_point
save_bps_data["basis_point_finger"] = basis_point_finger

IS_TIP = True
IS_REVERSE_AUGMENT = True
contain_name = None
all_data = Path(dataset_path).glob("**/*.pt")
all_data = list(all_data)
for path in tqdm(all_data):
    if contain_name is not None:
        if contain_name not in path.name:
            continue
    vid_name = path.parent.name + "_" + path.name
    # print(f"Loading {vid_name}")
    motion_data = load_arctic_data(path)
    humanoid_localmat, humanoid_globalmat = get_humanoid_data(motion_data)
    obj_localmat, obj_globalmat = get_obj_data(motion_data)
    angles = motion_data["obj"]["angles"]
    obj_name = path.name.split("_")[0]
    data_dict = process_bps(humanoid_globalmat.cuda(), obj_globalmat.cuda(), angles.cuda(), obj_name)
    print(f"Saving {vid_name}")
    save_bps_data[vid_name] = data_dict
    if IS_REVERSE_AUGMENT:
        reverse_humanoid_globalmat = humanoid_globalmat.clone().flip(dims=[0])
        reverse_obj_globalmat = obj_globalmat.clone().flip(dims=[0])
        reverse_angles = angles.clone().flip(dims=[0])
        reverse_data_dict = process_bps(
            reverse_humanoid_globalmat.cuda(), reverse_obj_globalmat.cuda(), reverse_angles.cuda(), obj_name
        )
        print(f"Saving {vid_name}_reverse")
        save_bps_data[vid_name + "_reverse"] = reverse_data_dict


torch.save(save_bps_data, f"./inputs/arctic_bps.pth")
