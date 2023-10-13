import os
import sys
import traceback
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def read_ecg_csv_files():
	np_data_directory = "data/cap-sleep-database-1.0.0/np_data"
	class_map={
		"S0": 0,
		"S1": 1,
		"S2": 2,
		"S3": 3,
		"S4": 3,
		"REM": 4,
	}
	seg_count = 0
	seg_sz = 30 * 50  # 30s x 50Hz
	reached_sleep_stage_line = False
	np_data_directory = f"{data_directory}/np_data"
	for f in os.listdir(np_data_directory):
		if not f.endswith(".csv"):
		        continue
		rec_name = f[:-4]
		sig_filepath = f"{np_data_directory}/{f}"
		ecg_sig = np.loadtxt(sig_filepath, delimiter=",")
		annot_filepath = sig_filepath.replace('.csv', '.txt')
		with open(annot_filepath, "r") as f:
		    lines = f.readlines()
		print(f"[{rec_name}] {ecg_sig.shape}")
		start = 0
		ignored_annots_bad_len, ignored_annots_bad_label = 0, 0
		print(f"recording:{ecg_sig.shape}")
		for line in lines:
		    # print(line)
		    tokens = line.split("\t")
		    if not reached_sleep_stage_line and not line.startswith("Sleep Stage"):
		        continue
		    if line.startswith("Sleep Stage"):
		        reached_sleep_stage_line = True
		        continue
		    try:
		        tok_is_sleep = tokens[-3]
		        if tok_is_sleep.find("SLEEP-") < 0:
		            continue
		        label_str = tok_is_sleep.split("-")[-1]
		        label = class_map.get(label_str)
		        current_epoch_len = int(tokens[-2])
		    except:
		        print(f"--> error parsing epoch-len: {tokens[-2]}")
		        ignored_annots_bad_len += 1
		        continue
		    if current_epoch_len != 30:  # not a valid sleep epoch
		        ignored_annots_bad_len += 1
		        print(f"--> bad length: {current_epoch_len}")
		        # start += (current_epoch_len*self.hz)
		        continue
		    # label_str = tokens[0].strip()
		    # label = class_map.get(label_str)
		    # if label is None:
		    #     print(
		    #         f"Ignored label for label_str:{label_str} -> {label}")
		    #     ignored_annots_bad_label += 1
		    #     start += seg_sz
		    #     continue
		    
		    seg = ecg_sig[start : start + seg_sz]
		    print(
		        f"recording:{ecg_sig.shape}, segment:{len(seg)}, {start} -> {start + seg_sz}")

		    if len(seg) != seg_sz:
		        print(f"bad seg_sz:{len(seg)}")
		        start += seg_sz
		        continue

		    # update start
		    start += seg_sz
		    seg_count += 1

		print(f"...[{rec_name}], n_seg:{seg_count}, bad_label: {ignored_annots_bad_label}, bad_len/ignored: {ignored_annots_bad_len}, ")
		break


def edf_to_csv(data_directory):
    # Find and read .edf files and corresponding .txt files for annotation.
    #
    count_file = 0
    ignore_file = 0
    bad_files = []
    record_names = []
    target_data_dir = f"{data_directory}/np_data"
    if not os.path.exists(target_data_dir):
        os.makedirs(target_data_dir)

    for f in os.listdir(data_directory):
        if not f.endswith(".edf"):
            continue
        rec_name = f[:-4]  # exclude .edf
        try:
            edf_path = f"{data_directory}/{f}"
            recording, _, _ = load_cap_ecg_data_mne(edf_path, fs_target=TARGET_HZ)
            print(
                f"{count_file} | {f} | signal:{recording.shape}")
            record_names.append(rec_name)
            np.savetxt(f"{target_data_dir}/{f.replace('.edf', '.csv')}", recording)

            os.system(f"cp {data_directory}/{f.replace('.edf', '.txt')} {target_data_dir}")
            # rr_signal = generate_rr_signal(recording)
            # print(f"rr_signal")
            # break
        except:
            ignore_file += 1
            print(f"error loading {f}")
            traceback.print_exc(file=sys.stdout)
            bad_files.append(f)
            # Ignore corrupted recording.
            continue

        count_file += 1    
    print(f"{count_file} files created.")
    
    
def load_cap_ecg_data_mne(edf_file, fs_target=50, log=print):
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False)
        ch_idx = -1
        ch_name = None
        for cname in raw.info.get('ch_names'):
            ch_idx += 1
            if cname.find("ECG") > -1:
                ch_name = cname
                break
        hz = mne.pick_info(raw.info, [ch_idx], verbose=False)['sfreq']
        hz = int(hz)
        raw.pick_channels([ch_name])
        recording = raw.get_data().flatten()
        print(f"channel: {cname}")

    except:
        log(f"Error reading {edf_file}, caused by - {traceback.format_exc()}")
        return
    meta = {
        "hz": hz
    }
    # Down sample to 50Hz
    down_factor = meta["hz"] // fs_target
    target_samples = len(recording) // down_factor
    print(f"{edf_file}, @{meta['hz']}, down_factor:{down_factor}, recording:{recording.shape}, target:{target_samples}")
    recording = signal.resample(recording, target_samples)

    # Read annotation
    labels = []
    return (recording, labels, meta)


def main():
	read_ecg_csv_files()
	
	# TARGET_HZ = 50
	# data_directory = "data/cap-sleep-database-1.0.0"
	# edf_to_csv(data_directory)


if __name__ == "__main__":
	main()

