import torch
import einops
import torch.nn as nn


def cape_embed(f, P, is_query=False):
    """
    Apply CaPE on feature.
    :param f: feature vector of shape [..., d]
    :param P: 4x4 transformation matrix
    :return: rotated feature f by pose P: f@P
    """
    f = einops.rearrange(f, "... (d k) -> ... d k", k=4)  # (B, N, L, C) -> (B, N, L, C//4, 4)
    if is_query:
        P = torch.inverse(P).permute(0, 1, 3, 2)  # we need transpose here as key would be transposes in attention
    P = einops.repeat(P, "b t m n -> b l t m n", l=f.shape[1])
    return einops.rearrange(f @ P, "... d k -> ... (d k)", k=4)


########################## 6DoF CaPE ####################################
# class CaPE_6DoF:
#     def cape_embed(self, f, P):
#         """
#         Apply CaPE on feature.
#         :param f: feature vector of shape [..., d]
#         :param P: 4x4 transformation matrix
#         :return: rotated feature f by pose P: f@P
#         """
#         f = einops.rearrange(f, '... (d k) -> ... d k', k=4)
#         return einops.rearrange(f@P, '... d k -> ... (d k)', k=4)

#     def attn_with_CaPE(self, f1, f2, p1, p2):
#         """
#         Do attention dot production with CaPE pose encoding.
#         # query = cape_embed(query, p_out_inv)  # query f_q @ (p_out)^(-T)
#         # key = cape_embed(key, p_in)  # key f_k @ p_in
#         :param f1: b (t1 l) d
#         :param f2: b (t2 l) d
#         :param p1: [b, t, 4, 4]
#         :param p2: [b, t, 4, 4]
#         :return: attention score: q@k.T
#         """
#         l = f1.shape[1] // p1.shape[1]
#         assert f1.shape[1] // p1.shape[1] == f2.shape[1] // p2.shape[1]
#         p1_invT = einops.repeat(torch.inverse(p1).permute(0, 1, 3, 2), 'b t m n -> b (t l) m n', l=l)  # f1 [b, l*t1, d]
#         query = self.cape_embed(f1, p1_invT)  # [b, l*t1, d] query: f1 @ (p1)^(-T), transpose the last two dim
#         p2_copy = einops.repeat(p2, 'b t m n -> b (t l) m n', l=l) # f2 [b, l*t2, d]
#         key = self.cape_embed(f2, p2_copy)  # [b, l*t2, d] key: f2 @ p2
#         att = query @ key.permute(0, 2, 1)  # [b, l*t1, l*t2] attention: query@key^T
#         return att


# ################### 6DoF Verification ###################################
# def euler_to_matrix(alpha, beta, gamma, x, y, z):
#     # radian
#     r = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)
#     t = np.array([[x], [y], [z]])
#     rot_matrix = r.as_matrix()
#     rot_matrix = np.concatenate([rot_matrix, t], axis=-1)
#     rot_matrix = np.concatenate([rot_matrix, [[0, 0, 0, 1]]], axis=0)
#     return rot_matrix

# def random_6dof_pose(B, T):
#     pose_euler = torch.rand([B, T, 6]).numpy()  # euler
#     pose_matrix = []
#     for b in range(B):
#         p = []
#         for t in range(T):
#             p.append(torch.from_numpy(euler_to_matrix(*pose_euler[b, t])))
#         pose_matrix.append(torch.stack(p))
#     pose_matrix = torch.stack(pose_matrix)
#     return pose_matrix.float()

# bs = 6  # batch size
# t1 = 3  # num of target views in each batch, can be arbitrary number
# t2 = 5  # num of reference views in each batch, can be arbitrary number
# l = 10  # len of token
# d = 16  # dim of token feature, need to mod 4 in this case
# assert d % 4 == 0

# # random init query and key
# f1 = torch.rand(bs, t1, l, d)     # query
# f2 = torch.rand(bs, t2, l, d)     # key
# f1 = einops.rearrange(f1, 'b t l d -> b (t l) d')
# f2 = einops.rearrange(f2, 'b t l d -> b (t l) d')

# # random init pose p1, p2, delta_p, [bs, t, 4, 4]
# p1 = random_6dof_pose(bs, t1)   # [bs, t1, 4, 4]
# p2 = random_6dof_pose(bs, t2)   # [bs, t2, 4, 4]
# p_delta = random_6dof_pose(bs, 1)   # [bs, 1, 4, 4]
# # delta p is identical to p1 and p2 in each batch
# p1_delta = einops.repeat(p_delta, 'b 1 m n -> b (1 t) m n', t=t1//1)
# p2_delta = einops.repeat(p_delta, 'b 1 m n -> b (1 t) m n', t=t2//1)

# # run attention with CaPE 6DoF
# cape_6dof = CaPE_6DoF()
# # att
# att = cape_6dof.attn_with_CaPE(f1, f2, p1, p2)
# # att_delta
# att_delta = cape_6dof.attn_with_CaPE(f1, f2, p1@p1_delta, p2@p2_delta)

# # condition: att score should be the same i.e. non effect from any delta_p
# assert torch.allclose(att, att_delta, 1e-3)
# print("6DoF CaPE Verified")


class PointRoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        assert d_model % 6 == 0

    def rotate_every_two(self, x):
        x = einops.rearrange(x, "... (d j) -> ... d j", j=2)  # (B, N, M, C) -> (B, N, M, C//2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)  # (B, N, M, C//2, 2) -> (B, N, M, C//2, 2)
        return einops.rearrange(x, "... d j -> ... (d j)")  # (B, N, M, C//2, 2) -> (B, N, M, C)

    def cape(self, x, p):
        d, m, n = x.shape[-1], p.shape[-2], p.shape[-1]
        assert d % (2 * n) == 0
        emb = einops.repeat(p, "... n -> ... (n k)", k=d // n)  # (B, N, M, 3) => (B, N, M, C)
        return emb

    def cape_embed(self, qq, kk, p1, p2):
        """
        Embed camera position encoding into attention map
        :param qq: query feature map   [b, n, l_q, feature_dim]
        :param kk: key feature map    [b, n, l_k, feature_dim]
        :param p1: query pose  [b, l_q, pose_dim]
        :param p2: key pose    [b, l_k, pose_dim]
        :return: cape embedded attention map    [b, n, l_q, l_k]
        """

        p1 = einops.repeat(p1, "b l d -> b n l d", n=qq.shape[1])
        p2 = einops.repeat(p2, "b l d -> b n l d", n=kk.shape[1])

        m1 = self.cape(qq, p1)
        m2 = self.cape(kk, p2)

        q = (qq * m1.cos()) + (self.rotate_every_two(qq) * m1.sin())
        k = (kk * m2.cos()) + (self.rotate_every_two(kk) * m2.sin())

        return q, k
