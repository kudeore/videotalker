import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
import gc # For garbage collection

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                 trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")

args = options()

# --- Global/Cached resources (initialized once) ---
_enhancer = None
_restorer = None
_croper = None
_kp_extractor_3dmm = None # For initial 3dmm extraction
_net_recon = None
_D_Net = None
_model = None
_ganimation_instance = None
_lm3d_std = None

def get_global_resources(device):
    global _enhancer, _restorer, _croper, _kp_extractor_3dmm, _net_recon, _D_Net, _model, _ganimation_instance, _lm3d_std

    if _enhancer is None:
        _enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                                   sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    if _restorer is None:
        _restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean',
                            channel_multiplier=2, bg_upsampler=None)
    if _croper is None:
        _croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    if _kp_extractor_3dmm is None:
        _kp_extractor_3dmm = KeypointExtractor() # Used for initial 3DMM extraction
    if _net_recon is None:
        _net_recon = load_face3d_net(args.face3d_net_path, device)
    if _lm3d_std is None:
        _lm3d_std = load_lm3d('checkpoints/BFM')
    if _D_Net is None or _model is None:
        _D_Net, _model = load_model(args, device)
    if args.up_face != 'original' and _ganimation_instance is None:
        _ganimation_instance = GANimationModel()
        _ganimation_instance.initialize()
        _ganimation_instance.setup()

    return _enhancer, _restorer, _croper, _kp_extractor_3dmm, _net_recon, _D_Net, _model, _ganimation_instance, _lm3d_std

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)

    # Initialize all heavy models once
    enhancer, restorer, croper, kp_extractor_3dmm, net_recon, D_Net, model, ganimation_instance, lm3d_std = get_global_resources(device)

    base_name = os.path.splitext(os.path.basename(args.face))[0] # Get base name without extension

    # --- Video Input Setup ---
    if os.path.isfile(args.face) and args.face.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Handle static image case (already done, but emphasize it's not streaming)
        print("[Info] Processing a static image. This will not be streamed.")
        # Original logic for static image:
        full_frames_single = [cv2.imread(args.face)]
        fps = args.fps
        # Process this single frame through the existing pipeline (or adapt it)
        # For simplicity, we'll assume the rest of the pipeline can handle a single frame
        # For high-res/fps, this branch is less relevant.
        print("Static image processing is not optimized for high-res/fps in this example.")
        return # Exit for static image for this refactored example
    else:
        video_stream = cv2.VideoCapture(args.face)
        if not video_stream.isOpened():
            raise ValueError(f"Could not open video file: {args.face}")
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Info] Input video FPS: {fps}, Total frames: {total_frames}")

    # --- Audio Processing (can be done once or chunked if very long) ---
    audio_path = args.audio
    if not audio_path.endswith('.wav'):
        temp_wav_path = os.path.join('temp', args.tmp_dir, 'temp.wav')
        command = f'ffmpeg -loglevel error -y -i {audio_path} -strict -2 {temp_wav_path}'
        subprocess.call(command, shell=True)
        audio_path = temp_wav_path
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Add a small epsilon noise to the wav file and try again')

    mel_step_size, mel_idx_multiplier = 16, 80. / fps
    # Pre-calculate all mel chunks (audio is relatively small)
    mel_chunks = []
    for i in range(total_frames): # Iterate based on video frames
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
        else:
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
    print(f"[Step 4] Load audio; Length of mel chunks: {len(mel_chunks)}")

    # --- Prepare Output Video Writer ---
    frame_h, frame_w = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_video_path = os.path.join('temp', args.tmp_dir, 'result_temp.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    # --- Main Streaming Loop ---
    # We need to maintain some state across frames for stabilization and alignment
    # Store initial crop and quad from the first frame
    first_frame_read, first_frame = video_stream.read()
    if not first_frame_read:
        raise ValueError("Could not read first frame of video.")

    # Apply initial crop if specified
    y1, y2, x1, x2 = args.crop
    if x2 == -1: x2 = first_frame.shape[1]
    if y2 == -1: y2 = first_frame.shape[0]
    first_frame_cropped = first_frame[y1:y2, x1:x2]

    first_frame_RGB = cv2.cvtColor(first_frame_cropped, cv2.COLOR_BGR2RGB)
    full_frames_RGB_cropped, crop_coords, quad_coords = croper.crop([first_frame_RGB], xsize=512)
    initial_full_frame_RGB_cropped = full_frames_RGB_cropped[0] # The 512x512 cropped face
    initial_crop_coords = crop_coords # (clx, cly, crx, cry)
    initial_quad_coords = quad_coords # (lx, ly, rx, ry)

    # Calculate original face region in the full frame based on initial crop
    clx, cly, crx, cry = initial_crop_coords
    lx, ly, rx, ry = initial_quad_coords[0]
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, first_frame.shape[0]), clx+lx, min(clx+rx, first_frame.shape[1])
    initial_face_region_in_full_frame = (oy1, oy2, ox1, ox2)

    # Extract 3DMM for the first frame (reference)
    first_frame_pil = Image.fromarray(cv2.resize(initial_full_frame_RGB_cropped, (256, 256)))
    lm_first = kp_extractor_3dmm.extract_keypoint([first_frame_pil], 'temp/'+base_name+'_first_frame_landmarks.txt')[0]
    if np.mean(lm_first) == -1:
        lm_first = (lm3d_std[:, :2]+1) / 2.
        lm_first = np.concatenate([lm_first[:, :1] * 256, lm_first[:, 1:2] * 256], 1)
    else:
        lm_first[:, -1] = 256 - 1 - lm_first[:, -1]
    trans_params_first, im_idx_first, lm_idx_first, _ = align_img(first_frame_pil, lm_first, lm3d_std)
    trans_params_first = np.array([float(item) for item in np.hsplit(trans_params_first, 5)]).astype(np.float32)
    im_idx_tensor_first = torch.tensor(np.array(im_idx_first)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
    with torch.no_grad():
        coeffs_first = split_coeff(net_recon(im_idx_tensor_first))
    semantic_first_frame_numpy = np.concatenate([coeffs_first['id'], coeffs_first['exp'], coeffs_first['tex'],
                                                 coeffs_first['angle'], coeffs_first['gamma'], coeffs_first['trans'],
                                                 trans_params_first[None]], 1)[0]
    del im_idx_tensor_first, coeffs_first # Release memory

    # Determine expression
    expression = None
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from',args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        W, H = exp_pil.size
        lm_exp = kp_extractor_3dmm.extract_keypoint([exp_pil], 'temp/'+base_name+'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate([lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]
        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del im_exp_tensor # Release memory
    elif args.exp_img == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # Release net_recon if not needed anymore, or keep if you plan to re-extract 3DMM on other frames
    # For high-res/fps, it's better to reuse semantic_first_frame_numpy and only extract 3DMM on keyframes or not at all
    # if the head pose is stable. For this refactor, we assume a stable head pose and use first frame's coeffs.
    del net_recon # No longer need 3DMM network after first frame extraction
    torch.cuda.empty_cache()
    gc.collect() # Explicitly call garbage collector

    # --- Process Frame by Frame ---
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="[Step 6] Lip Synthesis (Streaming)")

    # Rewind video stream to start (as we read the first frame already)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break

        # Apply initial crop to the current frame
        current_frame_cropped_original_res = frame[y1:y2, x1:x2]
        current_frame_RGB_cropped_original_res = cv2.cvtColor(current_frame_cropped_original_res, cv2.COLOR_BGR2RGB)

        # Resize for face processing (e.g., 256x256 or 512x512)
        face_img_for_processing = cv2.resize(current_frame_RGB_cropped_original_res, (256, 256))
        face_img_pil = Image.fromarray(face_img_for_processing)

        # --- Step 1: Face Detection & Cropping (using the pre-calculated crop/quad from first frame) ---
        # For high-res/fps, we assume the face is generally in the same region,
        # and use the initial crop/quad for efficiency.
        # If the head moves significantly, you'd need to re-run face detection per frame/chunk.
        
        # We need the current frame's face region for enhancement later.
        # The original `face_detect` function is used here.
        current_face_det_results = face_detect([frame], args, jaw_correction=True)[0]
        oface_current, coords_current = current_face_det_results
        
        # Resize `oface_current` and `face_img_for_processing` to `args.img_size` (384)
        oface_current_resized = cv2.resize(oface_current, (args.img_size, args.img_size))
        face_img_for_processing_resized = cv2.resize(face_img_for_processing, (args.img_size, args.img_size))

        # --- Step 2 & 3: 3DMM Extraction & Stabilization (Simplified for streaming) ---
        # For high-res/fps, re-extracting 3DMM for every frame is too slow.
        # We'll use the semantic coefficients from the first frame as a base,
        # and only modify the expression component.
        
        # Use semantic_first_frame_numpy as the base semantic for all frames
        semantic_current_numpy = semantic_first_frame_numpy[None, :] # Add batch dimension
        
        # Ratio calculation might still be needed if there's slight movement, but simplified
        # For a truly static head, ratio would be 1.0. If head moves, this needs re-evaluation.
        # For simplicity, assuming ratio is derived from semantic_first_frame_numpy vs itself.
        ratio = find_crop_norm_ratio(semantic_current_numpy, semantic_current_numpy) # simplified
        coeff = transform_semantic(semantic_current_numpy, 0, ratio).unsqueeze(0).to(device) # idx=0 as we use the base semantic

        # Hacking the new expression
        coeff[:, :64, :] = expression[None, :64, None].to(device)

        # Prepare source image for D_Net
        source_img = trans_image(face_img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = D_Net(source_img, coeff)
        img_stablized_raw = (output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255
        img_stablized = cv2.cvtColor(np.uint8(img_stablized_raw), cv2.COLOR_RGB2BGR)

        # --- Step 5: Reference Enhancement (on the stabilized face) ---
        pred_enhanced, _, _ = enhancer.process(img_stablized, img_stablized, face_enhance=True, possion_blending=False)

        # --- Step 6: Lip Synthesis (using the batching mechanism of datagen conceptually) ---
        # Instead of `datagen` generating all batches, we feed one frame/mel at a time
        
        # Prepare inputs for model (LNet and ENet)
        # img_batch: concatenated masked original face and reference face
        # mel_batch: current mel chunk
        
        # Create masked image (half of the face)
        img_masked = oface_current_resized.copy()
        img_masked[:, args.img_size//2:] = 0 # Mask out lower half of face
        
        # Reference face (from the current frame's detected face)
        ref_face = face_img_for_processing_resized # Using the 256x256 processed face as reference

        img_batch_input = np.concatenate((img_masked, ref_face), axis=2) / 255.0 # Concatenate along width
        mel_batch_input = np.reshape(mel_chunks[frame_idx], [1, mel_chunks[frame_idx].shape[0], mel_chunks[frame_idx].shape[1], 1])

        img_batch_tensor = torch.FloatTensor(np.transpose(img_batch_input, (2, 0, 1))).unsqueeze(0).to(device)
        mel_batch_tensor = torch.FloatTensor(np.transpose(mel_batch_input, (0, 3, 1, 2))).to(device)
        img_original_tensor = torch.FloatTensor(np.transpose(oface_current_resized, (2, 0, 1))).unsqueeze(0).to(device) / 255.0 # BGR -> RGB already handled by cvtColor

        with torch.no_grad():
            incomplete, reference = torch.split(img_batch_tensor, 3, dim=1)
            pred, low_res = model(mel_batch_tensor, img_batch_tensor, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original_tensor * 2 - 1), size=(128, 128), mode='bilinear'),
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                ganimation_instance.feed_batch(test_batch)
                ganimation_instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(ganimation_instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')
            else:
                cur_gen_faces = img_original_tensor

            if args.without_rl1 is not False:
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete))
                pred = pred * mask + cur_gen_faces * (1 - mask)

        pred_lip_synced = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        pred_lip_synced = pred_lip_synced[0].astype(np.uint8) # Get the single frame

        torch.cuda.empty_cache()
        gc.collect() # Explicitly call garbage collector

        # --- Paste back to original frame & Final Enhancement ---
        # `coords_current` are the bounding box for the current frame's detected face
        y1_face, y2_face, x1_face, x2_face = coords_current
        
        # Resize the lip-synced prediction to the detected face size
        p_resized_to_face = cv2.resize(pred_lip_synced, (x2_face - x1_face, y2_face - y1_face))
        
        # Create a copy of the original full frame to paste onto
        current_full_frame_copy = frame.copy()
        current_full_frame_copy[y1_face:y2_face, x1_face:x2_face] = p_resized_to_face

        # Month region enhancement by GFPGAN
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            current_full_frame_copy, has_aligned=False, only_center_face=True, paste_back=True)

        mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0] # Mouth mask
        mouse_mask = np.zeros_like(restored_img)
        tmp_mask = enhancer.faceparser.process(restored_img[y1_face:y2_face, x1_face:x2_face], mm)[0]
        mouse_mask[y1_face:y2_face, x1_face:x2_face]= cv2.resize(tmp_mask, (x2_face - x1_face, y2_face - y1_face))[:, :, np.newaxis] / 255.0

        height, width = current_full_frame_copy.shape[:2]
        restored_img_resized, current_full_frame_copy_resized, full_mask_resized = [cv2.resize(x, (512, 512)) for x in (restored_img, current_full_frame_copy, np.float32(mouse_mask))]
        img_blended = Laplacian_Pyramid_Blending_with_mask(restored_img_resized, current_full_frame_copy_resized, full_mask_resized[:, :, 0], 10)
        final_frame_output = np.uint8(cv2.resize(np.clip(img_blended, 0 ,255), (width, height)))

        # Final face enhancement (GPEN)
        final_frame_output, orig_faces, enhanced_faces = enhancer.process(final_frame_output, frame, bbox=coords_current, face_enhance=False, possion_blending=True)
        out.write(final_frame_output)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    video_stream.release()
    out.release()

    # --- Final Audio-Video Merge ---
    final_output_filepath = args.outfile
    if not os.path.isdir(os.path.dirname(final_output_filepath)):
        os.makedirs(os.path.dirname(final_output_filepath), exist_ok=True)

    command = f'ffmpeg -loglevel error -y -i "{audio_path}" -i "{output_video_path}" -strict -2 -q:v 1 "{final_output_filepath}"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('outfile:', final_output_filepath)

    # Clean up temporary video file
    os.remove(output_video_path)
    if audio_path.startswith(os.path.join('temp', args.tmp_dir)): # Only remove if it was a temp wav
        os.remove(audio_path)

# Removed datagen function as its logic is now integrated into the main loop.
# The original datagen was designed to prepare all data upfront, which is not suitable for streaming.

if __name__ == '__main__':
    main()
