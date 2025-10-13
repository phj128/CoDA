import numpy as np
import torch
import trimesh


def intersect(obj_mesh, hand_mesh, engine="auto"):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def get_sample_intersect_volume(sample_info, mode="voxels"):
    hand_mesh = trimesh.Trimesh(vertices=sample_info["sbj_verts"], faces=sample_info["sbj_faces"])
    obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts"], faces=sample_info["obj_faces"])
    if mode == "engines":
        try:
            # sudo apt install openscad
            intersection = intersect(obj_mesh, hand_mesh, engine="scad")
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine="blender")
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            # the scad engine throws an exception if there is no intersection
            intersection = intersect(obj_mesh, hand_mesh, engine="blender")
            if intersection.is_empty:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
    elif mode == "voxels":
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume


def compute_max_depth(obj_faces, obj_vertices, sbj_vertices, penetr_mask):
    """
    Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric/penetration.py
    """
    obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
    trimesh.repair.fix_normals(obj_mesh)

    if penetr_mask.sum() == 0:
        max_depth = 0.0
        dist_to_closets_point_on_obj = None
    else:
        dist_to_closets_point_on_obj = _pre_compute_closest_dist(obj_faces, obj_vertices, sbj_vertices)
        max_depth = dist_to_closets_point_on_obj[penetr_mask == 1].max()

    return max_depth, dist_to_closets_point_on_obj


def _pre_compute_closest_dist(obj_faces, obj_vertices, sbj_vertices):
    # M, 3
    # N, 3
    obj_mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
    trimesh.repair.fix_normals(obj_mesh)
    (
        _,
        _dist_to_closets_point_on_obj,
        _,
    ) = trimesh.proximity.closest_point(obj_mesh, sbj_vertices)
    return _dist_to_closets_point_on_obj


def compute_interpenetration_volume_depth_mesh_2_mesh(obj_faces, obj_vertices, sbj_faces, sbj_vertices, penetr_mask):
    """
    Original source: https://github.com/hwjiang1510/GraspTTA/tree/master/metric
    https://github.com/CGAL/cgal-swig-bindings
    """
    # if do_intersect_single_frame_cgal(sbj_vertices, sbj_faces, obj_vertices, obj_faces):
    volume = get_sample_intersect_volume(
        sample_info={
            "sbj_verts": sbj_vertices,
            "obj_verts": obj_vertices,
            "sbj_faces": sbj_faces,
            "obj_faces": obj_faces,
        },
        mode="voxels",  # voxels
    )
    if volume is None:
        volume = 0.0
        max_depth = 0.0
    else:
        float(volume * 1e6)

    max_depth, dist_to_closets_point_on_obj = compute_max_depth(obj_faces, obj_vertices, sbj_vertices, penetr_mask)
    # else:
    #     volume = 0.0
    #     max_depth = 0.0

    return max_depth, volume, dist_to_closets_point_on_obj


def ray_triangle_intersection(ray_near, ray_dir, v123):
    """
    Möller–Trumbore intersection algorithm in pure python
    Based on http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    v1, v2, v3 = v123
    eps = 0.000001
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_dir, edge2)
    det = edge1.dot(pvec)
    retval = None
    ret = True
    if abs(det) < eps:
        ret = False
    inv_det = 1.0 / det
    tvec = ray_near - v1
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        ret = False
    qvec = np.cross(tvec, edge1)
    v = ray_dir.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        ret = False

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        ret = False
    retval = t
    if ret:
        print("v0", v1)
        print("v1", v2)
        print("v2", v3)
        print("edge1", edge1)
        print("edge2", edge2)
        print("det", det)
        print("tvec", tvec)
        print("u", u)
        print("v", v)
        print("qvec", qvec)
        print("pvec", pvec)

    return ret, retval


def batch_mesh_contains_points(
    ray_origins, obj_triangles, direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745])
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3).to(ray_origins.device)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)).view(
        batch_size, triangle_nb
    )

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = ~parallel
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior


def compute_contact_ratio(
    obj_faces, obj_vertices, sbj_vertices, dist_to_closets_point_on_obj=None, in_contact_threshold=0.005
):
    """
    nr. sbj vertices / nr. sbj vertices close to object (below in_contact_threshold)
    """
    n_sbj_vertices = sbj_vertices.shape[0]

    if dist_to_closets_point_on_obj is None:
        dist_to_closets_point_on_obj = _pre_compute_closest_dist(obj_faces, obj_vertices, sbj_vertices)

    n_verts_in_contact = np.sum(dist_to_closets_point_on_obj < in_contact_threshold)

    ratio = n_verts_in_contact / n_sbj_vertices

    return ratio


def build_sdf_grid(mesh, grid_dim=64, center=None, scale=None, device="cpu"):
    """
    构建SDF体素场，返回 [D, D, D] 的torch tensor
    """
    lin = torch.linspace(-1, 1, grid_dim, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [D*D*D, 3]
    grid_points_world = grid_points
    if scale is not None:
        grid_points_world = grid_points_world * scale.cpu()
    if center is not None:
        grid_points_world = grid_points_world + center.cpu()

    query = trimesh.proximity.ProximityQuery(mesh)
    sdf = query.signed_distance(grid_points_world.cpu().numpy())  # [D*D*D]
    sdf = torch.from_numpy(sdf).float().to(device).reshape(grid_dim, grid_dim, grid_dim)
    return sdf


def query_sdf_grid(sdf_grid, query_points):
    """
    query_points: [N, 3] 世界坐标
    sdf_grid: [D, D, D]
    center, scale: tensor, shape [3]
    返回: [N]，每个点的SDF值
    """
    query_points_shape = query_points.shape
    query_points = query_points.reshape(-1, 3)
    query_points = query_points.to(sdf_grid.device)
    coords = query_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, N, 1, 1, 3]
    # grid_sample 需要 [B, C, D, H, W] 和 [B, N, 1, 1, 3]
    sdf_grid = sdf_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, D, D, D]
    sdf_val = torch.nn.functional.grid_sample(sdf_grid, coords, mode="bilinear", align_corners=True)
    return sdf_val.reshape(query_points_shape[:-1])
