import torch, os, sys
from PIL import Image

import math

sys.path.append("/mmu-vcg-hdd/caiyufei05/Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3

sys.path.append("/mmu-vcg-hdd/caiyufei05/DiffSynth-Studio")

from diffsynth.core import ModelConfig
from diffsynth import load_state_dict

from src.qwen_image_pipeline import QwenImagePipeline
from src.unified_dataset import UnifiedDataset

from openexr_numpy import imread, imwrite
import torch.nn.functional as F


def get_orbit_viewmat(z_c, angle_deg, axis='y'):
    """
    计算绕前方球心旋转的相机外参矩阵 (World-to-Camera)
    球心坐标：C = (0, 0, z_c/2)
    """
    r = z_c / 2.0
    theta = math.radians(angle_deg)
    R = torch.eye(3)
    
    # 假设遵循标准 OpenCV 相机坐标系约定：X右，Y下，Z前
    if axis == 'y':   # Yaw (左右)
        R[0, 0] = math.cos(theta)
        R[0, 2] = math.sin(theta)
        R[2, 0] = -math.sin(theta)
        R[2, 2] = math.cos(theta)
    elif axis == 'x': # Pitch (上下)
        R[1, 1] = math.cos(theta)
        R[1, 2] = -math.sin(theta)
        R[2, 1] = math.sin(theta)
        R[2, 2] = math.cos(theta)

    # 球心坐标
    C = torch.tensor([0.0, 0.0, r]).view(3, 1)
    
    # 计算新相机在世界坐标系下的平移 t = C - R * C
    t = C - R @ C

    # 构建 Camera-to-World (c2w) 矩阵
    c2w = torch.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3:] = t
    
    # 模型通常需要 World-to-Camera (w2c) 矩阵，因此求逆
    w2c = torch.linalg.inv(c2w)
    return w2c


if __name__ == '__main__':

    device = torch.device("cuda")
    path_3D="../Depth-Anything-3/model/DA3-GIANT-1.1" #"../Depth-Anything-3/model/DA3METRIC-LARGE"
    model_3D = DepthAnything3.from_pretrained(path_3D)
    model_3D = model_3D.to(device=device)
    export_3D_feat_layers="19,27,33,39" #"4,11,17,23" 
    export_3D_feat_layers = export_3D_feat_layers.split(",")
    export_3D_feat_layers = [int(s) for s in export_3D_feat_layers]
    
    merge_3D = True

    # data_split = "eval" #"8K"

    prope_dim_arrange= [64, 20, 20, 24] #[64, 32, 32]
    if len(prope_dim_arrange) == 4:
        add_depth = True
    else:
        add_depth = False

    metadata_path = '../DL3DV-10K/meta_view/medium.csv'
    base_path = "../DL3DV-10K/DL3DV-Evaluation/images"
    data_split = "medium_demo"
    # metadata_path = '../RealEstate10K/meta_view/medium.csv'
    # base_path = "../RealEstate10K/JiaHWang/Re10K/test"
    # data_split = "Re10K_medium"
    # metadata_path = '../Sekai/meta_view/hard.csv'
    # base_path = "../Sekai/Sekai-Real-Walking-HQ-test"
    # data_split = "Sekai_hard"
    print(data_split)

    dataset = UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=2,
            data_file_keys="image,edit_image".split(","),
            prope=True,
            mode="val",
            norm_scale=1.0,
            anno_src="vipe-DA3",
            add_depth=add_depth,
            # path_3D="../Depth-Anything-3/model/DA3-GIANT-1.1",
            # export_3D_feat_layers="39",
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=base_path,
                max_pixels=1048576,
                height=None,
                width=None,
                height_division_factor=16,
                width_division_factor=16,
            )
        )

    ckpt_path = "models/train/Qwen-Image-Edit_add_3D_self_attn_GIANT_6144_vipe_depth_DA3_1K_merge_3D_resume"
    ckpts = os.listdir(ckpt_path)

    ckpt = "step-2500.safetensors"

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=1)


    suffix = ckpt_path.split("Qwen-Image-Edit")[-1] + f"_{data_split}"

    if not os.path.exists(f"result/{ckpt.split('.')[0]}{suffix}"):
        os.mkdir(f"result/{ckpt.split('.')[0]}{suffix}")

    state_dict = load_state_dict(os.path.join(ckpt_path, ckpt))
    pipe.dit.load_state_dict(state_dict, strict=False)

    cnt = 0
    print(len(dataset))
    for data in dataloader:
        # for ckpt in ckpts:

            prompt = ["镜头视角转到指定位置"]

            print(data["edit_image"].size)
            image = data["edit_image"].resize((960, 528))

            edit_image =  data["edit_image"].resize((960, 528))
            target_image = data["image"].resize((960, 528))
            print(data["name"])
            print(os.path.join(ckpt_path, ckpt))

            feat_3D = model_3D.inference(
                [data["edit_image"].resize((960, 528))],                                                 # (1, 33, 60)
                export_feat_layers=export_3D_feat_layers,   # (1, 20, 36, 1536) H, W, C  (1, 20, 36, 1024) 960 528
                process_res=840,
            )
            feats = []
            for layer in export_3D_feat_layers:
                feats.append(torch.from_numpy(feat_3D.aux[f"feat_layer_{layer}"]))
            feat_3D = torch.cat(feats, dim=-1).to(dtype=torch.bfloat16, device=device) # (1, 20, 36, 1536) H, W, C
                
            if add_depth:
                depth = data["depth"].unsqueeze(0)
            else:
                depth = None

            # -----------------------------------------------------------------
            # 获取深度图中心点，用来定义球心距离
            # 假设 data["depth"] 形状为 [H, W] 或 [C, H, W]
            # -----------------------------------------------------------------
            depth_squeeze = data["depth"].squeeze()
            z_c = depth_squeeze[depth_squeeze.shape[0]//2, depth_squeeze.shape[1]//2].item()

            # 定义 4 个我们要生成的相对位姿
            # 配置格式: (轴, 角度) - x轴对应上下俯仰Pitch, y轴对应左右偏航Yaw
            pose_configs = {
                "top": ("x", 30),
                "bottom": ("x", -30),
                "left": ("y", -60),
                "right": ("y", 60)
            }

            generated_views = []

            # 分别根据4个位姿推理
            for pose_name, (axis, angle) in pose_configs.items():
                
                # 计算新的目标相机位姿
                target_viewmat = get_orbit_viewmat(z_c, angle, axis)
                
                # 对齐 device 和 dtype 以避免 Tensor 不匹配报错
                target_viewmat = target_viewmat.to(device=data["viewmats"].device, dtype=data["viewmats"].dtype)

                gen_img = pipe(
                    prompt, edit_image=edit_image, edit_image_auto_resize=False,
                    seed=0,
                    viewmats=target_viewmat.unsqueeze(0),
                    Ks=data["Ks"].unsqueeze(0),  # 内参保持不变
                    prope_dim_arrange=prope_dim_arrange,
                    add_attn=True,
                    add_3D=True,
                    feat_3D=feat_3D,
                    depth=depth,
                    merge_3D=merge_3D,
                    val=True,
                    num_inference_steps=40,
                    height=528, width=960,
                )
                generated_views.append(gen_img)

            # -----------------------------------------------------------------
            # 拼接图像: Source | Top(30) | Bottom(-30) | Left(-60) | Right(+60)
            # -----------------------------------------------------------------
            num_images = 1 + 4
            new_image = Image.new('RGB', (960 * num_images, 528), (255, 255, 255))

            x_offset = 0
            images_to_stitch = [edit_image] + generated_views
            
            for img in images_to_stitch:
                new_image.paste(img, (x_offset, (528 - img.height) // 2))  
                x_offset += img.width
                
            new_image.save(f"result/{ckpt.split('.')[0]}{suffix}/image_{ckpt.split('.')[0]}_{cnt}.png")

            cnt += 1