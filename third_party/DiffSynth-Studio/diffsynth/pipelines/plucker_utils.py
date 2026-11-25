# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch
import numpy as np

TACO_1080P_INTRINSICS = (
    (1376.7512207, 0.0, 967.68585205),
    (0.0, 1376.24035645, 526.81701660),
    (0.0, 0.0, 1.0),
)
TACO_1080P_SOURCE_WIDTH = 1920


def taco_1080p_intrinsics(orig_height, orig_width, device):
    """Return TACO 1080p intrinsics for either the original or a centered aspect crop."""
    K_orig = torch.tensor(TACO_1080P_INTRINSICS, dtype=torch.float32, device=device)
    orig_height, orig_width = int(orig_height), int(orig_width)
    aspect = orig_width / orig_height
    source_aspect = TACO_1080P_SOURCE_WIDTH / 1080
    if orig_height == 1080 and aspect < source_aspect:
        crop_left = (TACO_1080P_SOURCE_WIDTH - orig_width) / 2.0
        K_orig[0, 2] -= crop_left
    return K_orig


def pose_9d_to_transform(poses_9d):
    """
    将9D pose表示转换为4x4变换矩阵
    
    Args:
        poses_9d: [N, 9] 或 [N, J, 9] 9D pose表示 (6D rotation + 3D translation)
    
    Returns:
        transforms: [N, 4, 4] 或 [N, J, 4, 4] 变换矩阵
    """
    original_shape = poses_9d.shape
    if len(original_shape) == 3:  # [N, J, 9]
        N, J, _ = original_shape
        poses_9d = poses_9d.reshape(-1, 9)  # [N*J, 9]
    
    # 分离旋转和平移
    rot_6d = poses_9d[:, :6]  # [B, 6]
    trans = poses_9d[:, 6:9]  # [B, 3]
    
    # 6D旋转转3x3矩阵
    # 使用Gram-Schmidt正交化
    x_raw = rot_6d[:, 0:3]  # [B, 3]
    y_raw = rot_6d[:, 3:6]  # [B, 3]
    
    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True)  # [B, 3]
    z = torch.cross(x, y_raw, dim=1)  # [B, 3]
    z = z / torch.norm(z, dim=1, keepdim=True)  # [B, 3]
    y = torch.cross(z, x, dim=1)  # [B, 3]
    
    R = torch.stack([x, y, z], dim=2)  # [B, 3, 3]
    
    # 构建4x4变换矩阵
    B = R.shape[0]
    transforms = torch.eye(4, device=poses_9d.device).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
    transforms[:, :3, :3] = R
    transforms[:, :3, 3] = trans
    
    if len(original_shape) == 3:  # 恢复形状 [N, J, 4, 4]
        transforms = transforms.reshape(N, J, 4, 4)
    
    return transforms

def extract_camera_poses_from_saved(poses):
    """
    从保存的poses文件中提取相机位姿
    
    Args:
        poses [N, J*9+9] 的数据
                   前面是关节poses，最后9维是相对相机pose
    
    Returns:
        cam_poses: [N, 4, 4] 累积的相机位姿（世界坐标系到相机）
    """
    # poses = np.load(poses_path)  # [N, J*9+9]
    # poses = torch.from_numpy(poses).float()
    
    N = poses.shape[0]
    rel_cam_poses_9d = poses[:, -9:]  # [N, 9] 最后9维是相对相机pose
    
    # 转换为变换矩阵
    rel_cam_transforms = pose_9d_to_transform(rel_cam_poses_9d)  # [N, 4, 4]
    
    # 累积变换得到绝对相机位姿
    cam_poses = torch.zeros(N, 4, 4, device=poses.device)
    cam_poses[0] = torch.eye(4)  # 第一帧作为参考
    
    for i in range(1, N):
        # rel_cam_transforms[i] 是 cam[i-1] @ cam[i]^{-1}
        # 所以: cam[i] = cam[i-1] @ rel_cam_transforms[i]^{-1}
        cam_poses[i] = cam_poses[i-1] @ torch.linalg.inv(rel_cam_transforms[i])
    
    return cam_poses

def adjust_intrinsics_for_resize(K, orig_h, orig_w, new_h, new_w):
    """Adjust intrinsics for a full-frame, potentially anisotropic resize."""
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    scale = torch.eye(3, dtype=K.dtype, device=K.device)
    scale[0, 0] = scale_x
    scale[1, 1] = scale_y
    return scale @ K

def compute_plucker_rays(R_w2c, t_w2c, K, height, width, downsample_factor=1):
    """
    计算相机的 Plucker rays
    
    Args:
        R_w2c: [N, 3, 3] 世界到相机的旋转矩阵
        t_w2c: [N, 3] 世界到相机的平移向量
        K: [3, 3] 相机内参矩阵
        height: 图像高度
        width: 图像宽度
        downsample_factor: 降采样因子（默认1，不降采样）
    
    Returns:
        plucker_rays: [N, H//ds, W//ds, 6] Plucker 坐标表示的射线
    """
    N = R_w2c.shape[0]
    device = R_w2c.device
    
    # 应用降采样
    h = height // downsample_factor
    w = width // downsample_factor
    
    # 创建像素网格
    y, x = torch.meshgrid(
        torch.arange(0, height, downsample_factor, device=device, dtype=torch.float32),
        torch.arange(0, width, downsample_factor, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 像素坐标转齐次坐标 [H, W, 3]
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    
    # 计算相机坐标系下的射线方向
    K_inv = torch.linalg.inv(K)  # [3, 3]
    rays_cam = torch.einsum('ij,hwj->hwi', K_inv, pixels)  # [H, W, 3]
    rays_cam = rays_cam / torch.norm(rays_cam, dim=-1, keepdim=True)  # 归一化
    
    # 扩展到批次维度 [N, H, W, 3]
    rays_cam = rays_cam.unsqueeze(0).expand(N, -1, -1, -1)
    
    # 将射线方向转换到世界坐标系
    R_c2w = R_w2c.transpose(-1, -2)  # [N, 3, 3]
    t_c2w = -torch.einsum("nij,nj->ni", R_c2w, t_w2c)  # [N, 3] 相机中心在世界坐标系
    
    rays_world = torch.einsum('nij,nhwj->nhwi', R_c2w, rays_cam)  # [N, H, W, 3]
    
    # 计算 Plucker 坐标
    camera_center = t_c2w.view(N, 1, 1, 3).expand(-1, h, w, -1)  # [N, H, W, 3]
    moment = torch.cross(camera_center, rays_world, dim=-1)  # [N, H, W, 3]
    
    # 拼接成 6D Plucker 坐标 [N, H, W, 6]
    plucker_rays = torch.cat([rays_world, moment], dim=-1)
    
    return plucker_rays

def poses_to_plucker_rays(
    poses, 
    focal_length=600.,
    orig_height=540, 
    orig_width=720,
    target_height=544,  # 32的倍数
    target_width=736,   # 32的倍数
    downsample_factor=1,
    camera_intrinsics=None,
    # device='cuda'
):
    """
    从保存的poses文件生成Plucker rays
    
    Args:
        poses_path: poses .npy文件路径
        focal_length: 相机焦距
        orig_height, orig_width: 原始视频分辨率
        target_height, target_width: 目标分辨率（需要是32的倍数）
        downsample_factor: 降采样因子
        camera_intrinsics: optional [3, 3] intrinsic matrix in the original video coordinate system
        device: 计算设备
    
    Returns:
        plucker_rays: [N, H, W, 6] numpy数组
    """
    # 如果poses全是0，就是被drop了，返回一个全0的就可以
    if torch.all(poses == 0):
        return torch.zeros(6, poses.shape[0], target_height, target_width, dtype=torch.float32, device=poses.device)

    # 提取相机位姿
    device = poses.device
    if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
        cam_poses = poses.float()
    else:
        cam_poses = extract_camera_poses_from_saved(poses)  # [N, 4, 4]
    # cam_poses = cam_poses.to(device)

    # R_x = torch.eye(4).to(device=cam_poses.device, dtype=cam_poses.dtype)
    # R_x[0, 0] = -1
    # R_x[1, 1] = -1
    # cam_poses = R_x[None] @ cam_poses
    
    # 提取R和t
    R_w2c = cam_poses[:, :3, :3]  # [N, 3, 3]
    t_w2c = cam_poses[:, :3, 3]   # [N, 3]
    
    # 构建原始相机内参
    if camera_intrinsics is not None:
        K_orig = camera_intrinsics
        if not torch.is_tensor(K_orig):
            K_orig = torch.as_tensor(K_orig, dtype=torch.float32, device=device)
        else:
            K_orig = K_orig.to(device=device, dtype=torch.float32)
        if K_orig.ndim == 3:
            K_orig = K_orig[0]
        if K_orig.shape != (3, 3):
            raise ValueError(f"camera_intrinsics must have shape [3, 3], got {tuple(K_orig.shape)}")
    elif orig_height == 1080:
        K_orig = taco_1080p_intrinsics(orig_height, orig_width, device)
    else:
        K_orig = torch.tensor([
            [focal_length, 0, orig_width / 2],
            [0, focal_length, orig_height / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    # print(orig_height, orig_width, target_height, target_width, K_orig)
    
    # 调整内参以适应目标分辨率
    K_target = adjust_intrinsics_for_resize(
        K_orig, orig_height, orig_width, 
        target_height, target_width
    )
    
    # 计算 Plucker rays
    plucker_rays = compute_plucker_rays(
        R_w2c, t_w2c, K_target,
        target_height, target_width,
        downsample_factor=downsample_factor
    ).permute(3, 0, 1, 2) # [6, N, H, W]
    
    return plucker_rays


# 使用示例
if __name__ == '__main__':
    # 示例：从poses文件生成plucker rays
    poses_path = 'path/to/your/poses.npy'
    focal_length = 600.0  # 你的相机焦距
    
    plucker_rays = poses_to_plucker_rays(
        poses_path=poses_path,
        focal_length=focal_length,
        orig_height=540,
        orig_width=720,
        target_height=544,
        target_width=736,
        downsample_factor=4,  # 降采样4倍
        device='cuda'
    )
    
    print(f"Generated Plucker rays with shape: {plucker_rays.shape}")
    # 输出: Generated Plucker rays with shape: (N, 136, 184, 6)
