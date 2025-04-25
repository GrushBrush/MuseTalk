# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import cv2
import copy
import traceback # Import for error printing
# Assuming FaceParsing is correctly imported and works
try:
    from face_parsing import FaceParsing
    fp = FaceParsing() # Initialize FaceParsing model
except ImportError:
    print("Error: Could not import FaceParsing. Make sure the 'face-parsing' library is installed.")
    class DummyFaceParsing:
        def __call__(self, image): print("Warning: FaceParsing unavailable."); return None
    fp = DummyFaceParsing()
except Exception as e_fp:
    print(f"Error initializing FaceParsing: {e_fp}")
    class DummyFaceParsing:
        def __call__(self, image): print("Warning: FaceParsing failed init."); return None
    fp = DummyFaceParsing()


def get_crop_box(box, expand):
    """Calculates an expanded cropping box around a given bounding box."""
    # (Keep this function as provided previously)
    if box is None or len(box) != 4: return None, 0
    try: x, y, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    except (ValueError, TypeError): print(f"Warn: Non-numeric face_box: {box}"); return None, 0
    if x1 <= x or y1 <= y: print(f"Warn: Non-positive w/h face_box: {box}"); return None, 0
    x_c, y_c = (x + x1) / 2, (y + y1) / 2
    w, h = x1 - x, y1 - y
    s = max(0, int(max(w, h) / 2 * expand))
    x_s, y_s = int(round(x_c - s)), int(round(y_c - s))
    x_e, y_e = int(round(x_c + s)), int(round(y_c + s))
    crop_box = [x_s, y_s, x_e, y_e]
    return crop_box, s

def face_seg(image):
    """Performs face parsing using the FaceParsing model."""
    # (Keep this function as provided previously)
    if not isinstance(image, Image.Image):
        try: image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except Exception as e_conv: print(f"Error converting to PIL in face_seg: {e_conv}"); return None
    try:
        seg_image = fp(image)
        if seg_image is None: return None
        if seg_image.size != image.size: seg_image = seg_image.resize(image.size, Image.NEAREST)
        return seg_image
    except Exception as e_seg: print(f"ERROR during face_seg: {e_seg}"); traceback.print_exc(); return None


def get_image(image,face,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    """Original blending function (likely for visualization/testing, not real-time)."""
    # (Keep this function as is from the previous response if needed elsewhere)
    print("Warning: get_image function called - this is likely not intended for real-time path.")
    return image


# ============================================================================
# === get_image_prepare_material with ADDED Debug Prints =====================
# ============================================================================
def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    """Generates blurred mask and VALID CLAMPED crop box for blending."""
    x_s_orig, y_s_orig, x_e_orig, y_e_orig = None, None, None, None # For debug print
    try:
        if not isinstance(image, np.ndarray): print("Error GIPM: Input image not NumPy."); return None, None
        if face_box is None: print("Error GIPM: face_box is None."); return None, None

        # <<< PRINT INPUT FACE BOX >>>
        print(f">>> DEBUG GIPM: Input face_box = {face_box}")

        body = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        w_img, h_img = body.size # Use PIL size

        x, y, x1, y1 = face_box
        crop_box_list, s = get_crop_box(face_box, expand)
        if crop_box_list is None: print("Warning GIPM: get_crop_box failed."); return None, None

        x_s, y_s, x_e, y_e = crop_box_list
        x_s_orig, y_s_orig, x_e_orig, y_e_orig = x_s, y_s, x_e, y_e # Store originals

        # <<< PRINT INITIAL CROP BOX >>>
        print(f">>> DEBUG GIPM: Initial crop_box = {[x_s_orig, y_s_orig, x_e_orig, y_e_orig]}")

        # --- Clamp/Validate crop_box ---
        x_s_clamped, y_s_clamped = max(0, x_s), max(0, y_s)
        x_e_clamped, y_e_clamped = min(w_img, x_e), min(h_img, y_e)

        # <<< PRINT CLAMPED CROP BOX >>>
        print(f">>> DEBUG GIPM: Clamped crop_box = {[x_s_clamped, y_s_clamped, x_e_clamped, y_e_clamped]}")

        # --- Check if CLAMPED box is valid (positive width and height) ---
        if x_s_clamped >= x_e_clamped or y_s_clamped >= y_e_clamped:
            print(f"DEBUG GIPM CLAMP FAIL: Input face_box={face_box}, "
                  f"Initial crop_box={[x_s_orig, y_s_orig, x_e_orig, y_e_orig]}, "
                  f"Clamped crop_box={[x_s_clamped, y_s_clamped, x_e_clamped, y_e_clamped]}")
            return None, None # Return None for both if clamped box is invalid
        # --- End Check ---

        # --- Proceed using CLAMPED coordinates ---
        face_large = body.crop((x_s_clamped, y_s_clamped, x_e_clamped, y_e_clamped))
        ori_shape = face_large.size
        if ori_shape[0] == 0 or ori_shape[1] == 0:
            print(f"Debug GIPM: face_large crop zero dim: {ori_shape}")
            return None, None

        mask_image = face_seg(face_large)
        if mask_image is None:
            print("Warning GIPM: face_seg returned None.")
            return None, None

        # --- Process Mask (using clamped coords relative to crop) ---
        paste_x = x - x_s_clamped; paste_y = y - y_s_clamped
        paste_x1 = x1 - x_s_clamped; paste_y1 = y1 - y_s_clamped
        paste_x, paste_y = max(0, paste_x), max(0, paste_y)
        paste_x1, paste_y1 = min(ori_shape[0], paste_x1), min(ori_shape[1], paste_y1)
        if paste_x >= paste_x1 or paste_y >= paste_y1:
             print(f"Warning GIPM: Invalid paste region. fb={face_box}, clamped_cb={[x_s_clamped, y_s_clamped, x_e_clamped, y_e_clamped]}")
             return None, None

        mask_small = mask_image.crop((paste_x, paste_y, paste_x1, paste_y1))
        mask_image_processed = Image.new('L', ori_shape, 0)
        mask_image_processed.paste(mask_small, (paste_x, paste_y))

        width, height = mask_image_processed.size
        top_boundary = int(height * upper_boundary_ratio)
        modified_mask_image = Image.new('L', ori_shape, 0)
        if 0 <= top_boundary < height:
            modified_mask_image.paste(mask_image_processed.crop((0, top_boundary, width, height)), (0, top_boundary))
        else: modified_mask_image = mask_image_processed

        blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
        mask_np_gray = np.array(modified_mask_image)
        mask_array_blurred = cv2.GaussianBlur(mask_np_gray, (blur_kernel_size, blur_kernel_size), 0)

        # --- Return the mask and the VALID CLAMPED crop box ---
        final_crop_box = [x_s_clamped, y_s_clamped, x_e_clamped, y_e_clamped]

        # <<< PRINT FINAL RETURNED CROP BOX >>>
        print(f">>> DEBUG GIPM Return: Mask Shape={mask_array_blurred.shape}, Final CropBox={final_crop_box}")
        return mask_array_blurred, final_crop_box

    except Exception as e_gipm:
        print(f"!!!!! ERROR inside get_image_prepare_material !!!!!"); print(f" Input face_box: {face_box}")
        print(f" Input image shape: {image.shape if isinstance(image, np.ndarray) else 'Not NumPy'}")
        traceback.print_exc(); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); return None, None


# ============================================================================
# === get_image_blending (CPU version - keep as is for now) ================
# ============================================================================
def get_image_blending(image,face,face_box,mask_array,crop_box):
    # (Keep this function as is from the previous response if needed elsewhere)
    print("Warning: get_image_blending function called - this is likely not intended for real-time path.")
    return image
