import os
import torch
import copy
from tqdm import tqdm
from pathlib import Path
from coda.utils.bps import bps_gen_ball_inside, calculate_bps, get_pc_center, get_pc_scale
from coda.utils.wis3d_utils import make_wis3d, add_motion_as_lines, add_lines_as_mesh
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from coda.utils.grab.object_model import ObjectModel
from coda.dataset.arctic.utils import *
from coda.utils.grab.mesh import Mesh
from coda.utils.grab.utils import parse_npz, params2torch, to_cpu, params2cuda


def process_bps(humanoid_globalmat, obj_globalmat, seq_data, obj_name, is_reverse=False, is_mirror=False):
    left_finger_traj = get_finger_trajectory(humanoid_globalmat, is_right=False, istip=IS_TIP)  # (N, 5, 3)
    right_finger_traj = get_finger_trajectory(humanoid_globalmat, is_right=True, istip=IS_TIP)  # (N, 5, 3)
    wrist_traj = get_wrist_trajectory(humanoid_globalmat)  # (N, 2, 3)
    if is_mirror:
        wrist_traj = matrix.mirror_pos_yzplane(wrist_traj)
        wrist_traj = torch.cat([wrist_traj[..., [1], :], wrist_traj[..., [0], :]], dim=-2)
        right_finger_traj_ = matrix.mirror_pos_yzplane(left_finger_traj)
        left_finger_traj = matrix.mirror_pos_yzplane(right_finger_traj)
        right_finger_traj = right_finger_traj_
    finger_traj = torch.cat([wrist_traj, left_finger_traj, right_finger_traj], dim=-2)  # (N, 12, 3)

    obj_mesh = os.path.join(grab_path, seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    if obj_name not in obj_sampled_ind_dict:
        sampled_v, select_ind = sample_farthest_points(
            torch.tensor(obj_vtemp, dtype=torch.float32, device=DEVICE)[None], K=4096
        )  # (1, 4096, 3)
        obj_vtemp = sampled_v[0].cpu().detach().numpy()
        obj_sampled_ind_dict[obj_name] = select_ind[0].cpu().numpy()
    else:
        obj_vtemp = obj_vtemp[obj_sampled_ind_dict[obj_name]]
    obj_m = ObjectModel(v_template=obj_vtemp)
    obj_m = obj_m.to(DEVICE)

    grab_obj_data = {}
    grab_obj_data["transl"] = torch.zeros_like(matrix.get_position(obj_globalmat[:1, 0, :, :]))
    grab_obj_data["global_orient"] = torch.zeros_like(
        matrix.matrix_to_axis_angle(matrix.get_rotation(obj_globalmat[:1, 0, :, :]))
    )
    with torch.no_grad():
        obj_out = obj_m(**grab_obj_data)
    obj_verts = obj_out.vertices
    if is_mirror:
        obj_verts = matrix.mirror_pos_yzplane(obj_verts)

    center = get_pc_center(obj_verts[0])  # (3, )
    scale = get_pc_scale(obj_verts[0], multiply=1.25)  # (1, )

    idle_bps, _ = calculate_bps(basis_point[None], obj_verts)

    grab_obj_data = {}
    grab_obj_data["transl"] = matrix.get_position(obj_globalmat[..., 0, :, :])
    grab_obj_data["global_orient"] = matrix.matrix_to_axis_angle(matrix.get_rotation(obj_globalmat[..., 0, :, :]))
    with torch.no_grad():
        obj_out = obj_m(**grab_obj_data)
    obj_verts = obj_out.vertices
    if is_mirror:
        obj_verts = matrix.mirror_pos_yzplane(obj_verts)

    global_obj_center = matrix.get_position_from(center[None], obj_globalmat[..., 0, :, :])  # (N, 3)
    global_obj_center_mat = matrix.get_TRS(
        matrix.get_rotation(obj_globalmat[..., 0, :, :]), global_obj_center
    )  # (N, 4, 4)

    obj_local_verts = matrix.get_relative_position_to(obj_verts, global_obj_center_mat)  # (N, M, 3)
    scaled_obj_local_verts = obj_local_verts / scale
    bps_dist, bps_ind = calculate_bps(basis_point[None], scaled_obj_local_verts)

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
        "bps": bps_dist.cpu(),
        "scale": scale.cpu(),
        "center": center.cpu(),
        "idle_bps": idle_bps.cpu(),
    }
    return data_dict


DEVICE = "cuda"
# DEVICE = "cpu"
N_POINT = 256
basis_point = bps_gen_ball_inside(n_bps=N_POINT, random_seed=100)  # (M, 3)
basis_point = basis_point.to(DEVICE)
basis_point_finger = basis_point.clone()

dataset_path = "./inputs/grab_neutral"
grab_path = f"./inputs/grab_extracted"

obj_sampled_ind_dict = {}

save_bps_data = {}
save_bps_data["basis_point"] = basis_point
save_bps_data["basis_point_finger"] = basis_point_finger

IS_TIP = True
IS_REVERSE_AUGMENT = True
IS_MIRROR = False
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
    obj_name = path.name.split("_")[0]

    seq_data = parse_npz(os.path.join(grab_path, f"grab/{path.parent.name}", f"{path.name.replace('.pt', '.npz')}"))

    data_dict = process_bps(humanoid_globalmat.to(DEVICE), obj_globalmat.to(DEVICE), seq_data, obj_name)
    print(f"Saving {vid_name}")
    save_bps_data[vid_name] = data_dict
    if IS_MIRROR:
        mirror_data_dict = process_bps(
            humanoid_globalmat.to(DEVICE), obj_globalmat.to(DEVICE), seq_data, obj_name, is_mirror=True
        )
        print(f"Saving {vid_name}_mirror")
        save_bps_data[vid_name + "_mirror"] = mirror_data_dict
    if IS_REVERSE_AUGMENT:
        reverse_humanoid_globalmat = humanoid_globalmat.clone().flip(dims=[0])
        reverse_obj_globalmat = obj_globalmat.clone().flip(dims=[0])

        reverse_data_dict = process_bps(
            reverse_humanoid_globalmat.to(DEVICE), reverse_obj_globalmat.to(DEVICE), seq_data, obj_name, is_reverse=True
        )
        print(f"Saving {vid_name}_reverse")
        save_bps_data[vid_name + "_reverse"] = reverse_data_dict
        if IS_MIRROR:
            reverse_mirror_data_dict = process_bps(
                reverse_humanoid_globalmat.to(DEVICE),
                reverse_obj_globalmat.to(DEVICE),
                seq_data,
                obj_name,
                is_mirror=True,
                is_reverse=True,
            )
            print(f"Saving {vid_name}_reverse_mirror")
            save_bps_data[vid_name + "_reverse_mirror"] = reverse_mirror_data_dict

torch.save(save_bps_data, "./inputs/grab_bps.pth")
