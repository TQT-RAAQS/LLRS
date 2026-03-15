import pickle
import numpy as np
from experiment.toolkits.configs import Addresses
from experiment.toolkits.data_repository import *
import sys

for line in sys.stdin:
    command = line.strip()
    if command == "quit":
        break
    elif command == "reload_iqmixer":
        # Reload IQMixer parameters
        iqmixer_data = pickle.load(open(Addresses.iqmixer_wf_params_v, "rb"))
        dphi = iqmixer_data["dphi"]
        vdc_I = iqmixer_data["vdc_I"]
        vdc_Q = iqmixer_data["vdc_Q"]
        
        with open(Addresses.llrs_iqmixer_translation, "wb") as f:
            f.write(np.array([dphi, vdc_I, vdc_Q], dtype=np.float64).tobytes())
        
        with open(Addresses.llrs_iqmixer_translation_done, "w") as f:
            f.write("")
        
    elif command == "reload_psf":
        # Reload everything
        psf_data = pickle.load(open(Addresses.traps_psf, "rb"))
        thresholds_repo = GlobalDataRepository.get_data(DataLabel.THRESHOLDS)

        centers = psf_data['centers']   # shape (N, 2)
        psfs = psf_data['psfs']         # shape (N, w, h)
        N = centers.shape[0]
        w = psf_data['box_size_w']
        h = psf_data['box_size_h']
        image_count = thresholds_repo.shape[0]
        
        # Fetch cropping information
        cropping_data = psf_data["cropping"]
        is_cropping_active = cropping_data["is_cropping"]
        is_hardware_cropping = cropping_data["hardware"]
        cropping_center = cropping_data["cropping_center"]
        cropping_width = cropping_data["cropping_width"]
        cropping_height = cropping_data["cropping_height"]
        flag_center_shift_needed = is_cropping_active and not is_hardware_cropping
        
        # Shift centers if needed
        if flag_center_shift_needed:
            # Shift centers
            shift_x = cropping_center[1] - cropping_width // 2
            shift_y = cropping_center[0] - cropping_height // 2
            for i in range(len(centers)):
                centers[i][1] += shift_x  # y component
                centers[i][0] += shift_y  # x component

        # --- Write PSF binary file ---
        with open(Addresses.llrs_psfs_translation, "wb") as f:
            # Header: trap_count, box_size_w, box_size_h, image_count
            f.write(np.array([N, w, h, image_count], dtype=np.int64).tobytes())

            # For each trap: center (yc, xc) and PSF data
            for (yc, xc), p in zip(centers, psfs):
                f.write(np.array([yc, xc], dtype=np.int64).tobytes())
                f.write(np.array(p.ravel(), dtype=np.float64).tobytes())

            # Thresholds: shape (image_count, N)
            f.write(np.array(thresholds_repo, dtype=np.float64).tobytes())

        # --- Write trap orders binary file ---
        orders = psf_data["orders"]
        with open(Addresses.llrs_trap_orders_translation, "wb") as f:
            # Header: rows, cols
            f.write(np.array(orders.shape, dtype=np.int64).tobytes())
            # Data
            f.write(np.array(orders.ravel(), dtype=np.int64).tobytes())

        # Done file stays as text
        with open(Addresses.llrs_psfs_translation_done, "w") as f:
            f.write("")

    elif command == "reload_linear_controller_configs":
        # Reload linear controller configs
        linear_configs = pickle.load(open(Addresses.linear_controller_configs, "rb"))
        
        error_coefficients = linear_configs["error_coefficients"]  # shape (M,)
        correction_coefficients = linear_configs["correction_coefficients"]  # shape (M - 1,)
        M = len(error_coefficients)
        alpha = linear_configs["lp_alpha"]

        # --- Write linear controller configs binary file ---
        with open(Addresses.llrs_linear_controller_configs, "wb") as f:
            # Header: M, alpha
            f.write(np.array([M], dtype=np.int64).tobytes())
            f.write(np.array([alpha], dtype=np.float64).tobytes())
            # Coefficients
            f.write(np.array(error_coefficients, dtype=np.float64).tobytes())
            f.write(np.array(correction_coefficients, dtype=np.float64).tobytes())

        with open(Addresses.llrs_linear_controller_configs_done, "w") as f:
            f.write("")
