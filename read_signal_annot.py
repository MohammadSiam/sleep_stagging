import os
import sys
import traceback
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def read_ecg_csv_files():
	np_data_directory = "data/cap-sleep-database-1.0.0/np_data"
	class_map={
		"W": 0,
		"S1": 1,
		"S2": 2,
		"S3": 3,
		"S4": 3,
		"REM": 4,
		"R": 4,
	}
	seg_count = 0
	seg_sz = 30 * 50  # 30s x 50Hz
	reached_sleep_stage_line = False
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
		    #print(line)
		    tokens = line.split("\t")
		    if not reached_sleep_stage_line and not line.startswith("Sleep Stage"):
		        continue
		    if line.startswith("Sleep Stage"):
		        reached_sleep_stage_line = True
		        continue
		    try:
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
		    label_str = tokens[0].strip()
		    label = class_map.get(label_str)
		    if label is None:
		        print(
		            f"Ignored label for label_str:{label_str} -> {label}")
		        ignored_annots_bad_label += 1
		        start += seg_sz
		        continue
		    
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


def main():
	read_ecg_csv_files()


if __name__ == "__main__":
	main()

