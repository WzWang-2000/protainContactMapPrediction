import torch
import numpy as np

def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= (torch.norm(axis_x, dim=-1).unsqueeze(1) + eps)
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= (torch.norm(axis_z, dim=-1).unsqueeze(1) + eps)
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= (torch.norm(axis_y, dim=-1).unsqueeze(1) + eps)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec

def comp_feature(atoms):
    rotation = NormVec(atoms)
    r = torch.inverse(rotation)
    
    xyz_CA = torch.einsum('a b i, a i j -> a b j', atoms[:, 1].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    xyz_C  = torch.einsum('a b i, a i j -> a b j', atoms[:, 2].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    xyz_N  = torch.einsum('a b i, a i j -> a b j', atoms[:, 0].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    
    N_CA_C = torch.stack([xyz_N, xyz_CA, xyz_C], dim=-2)
    return N_CA_C

def read_from_pdb(file="1ry9A00_1.pdb") -> torch.tensor:
    """
    逐行读入pdb数据，将其中的N、CA、C的坐标提取出来，形成tensor
    :param file: PDB文件
    :return: key_atoms  # (L,3,3)
    """

    key_atoms_list = []
    i = -1  # 观察发现，N\CA\C以固定的顺序成对出现。引入i和k以保证这种顺序关系。
    N_list = []  # 记录N原子出现的行数

    with open(file, "r") as pdb:
        for atom_info in pdb:
            i += 1  # i记录了行数
            atom_list = atom_info.split()
            if "N" == atom_list[2]:
                key_atoms_list.append(atom_list[6:9])
                k = i
                N_list.append(k)  # 记录每个氨基酸的N出现的位置
            if "CA" == atom_list[2] and i == k + 1:
                key_atoms_list.append(atom_list[6:9])
            if "C" == atom_list[2] and i == k + 2:
                key_atoms_list.append(atom_list[6:9])

    L = len(N_list)

    key_atoms = np.reshape(np.array(key_atoms_list), [L, 3, 3]).astype("float32")
    key_atoms = torch.from_numpy(key_atoms)

    return key_atoms  # (L,3,3)

key_atoms = read_from_pdb("D:/anything/三轮轮转资料/pytest/1ry9A00.pdb")
key_atoms = torch.FloatTensor(key_atoms)
transform_xyz = comp_feature(key_atoms)
print(transform_xyz[0])