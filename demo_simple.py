import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
# from src.renderer.renderer import Renderer
from src.renderer.util import batch_orth_proj
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/test_image2.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    image_size = 224
    
    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        # checkpoint includes both smirk_encoder and smirk_generator
        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k}
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)

    image = cv2.imread(args.input_path)
    orig_image_height, orig_image_width, _ = image.shape

    kpt_mediapipe = run_mediapipe(image)

    # crop face if needed
    if args.crop:
        if (kpt_mediapipe is None):
            print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
            exit()
        
        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=image_size)
        
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

        cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
    else:
        cropped_image = image
        cropped_kpt_mediapipe = kpt_mediapipe

    output_image = cropped_image.copy()
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = cv2.resize(cropped_image, (224,224))
    cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
    cropped_image = cropped_image.to(args.device)

    outputs = smirk_encoder(cropped_image)
    flame_output = flame.forward(outputs)
    
    transformed_vertices = batch_orth_proj(flame_output['vertices'], outputs['cam'])
    flame_verts = transformed_vertices[0].detach().cpu()
    verts_total = int(flame_verts.size(0))
    
    obj_lines = []
    with open("assets/head_template.obj", "r") as obj_file:
        vert_counter = 0
        while True:
            line = obj_file.readline()
            
            if line.startswith("v ") and vert_counter < verts_total:
                obj_lines.append(f"v {flame_verts[vert_counter, 0]} {flame_verts[vert_counter, 1]} {flame_verts[vert_counter, 2]}\n")
                vert_counter += 1
            else:
                obj_lines.append(line)
            
            if not line:
                break
    
    os.makedirs(args.out_path, exist_ok=True)
    image_name = os.path.basename(args.input_path).split(".")[0]
    obj_name = f"{args.out_path}/{image_name}.obj"
    
    with open(obj_name, "w") as wf:
        wf.writelines(obj_lines)
    
    crop_name = f"{args.out_path}/{image_name}.png"
    cv2.imwrite(crop_name, output_image)