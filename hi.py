

# import numpy as np
# import os
# import sys

# # Attempt to import pymatreader for universal MATLAB v5/v7.3 support
# try:
#     from pymatreader import read_mat
#     HAS_PYMAT = True
# except ImportError:
#     import scipy.io as sio
#     HAS_PYMAT = False
#     print("[INFO] pymatreader not found, falling back to scipy.io")

# def inspect_file(path):
#     print(f"--- Inspecting: {path} ---")
#     if HAS_PYMAT:
#         try:
#             mat = read_mat(path)
#         except Exception as e:
#             print(f"[ERROR] pymatreader failed: {e}")
#             return
#     else:
#         mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)

#     if 'sentenceData' not in mat:
#         print("[ERROR] 'sentenceData' not found in file variables:", mat.keys())
#         return

#     sd = mat['sentenceData']
#     print(f"[INFO] Found 'sentenceData' with {len(sd)} sentences.")
    
#     # Inspect first sentence
#     s1 = sd[0]
#     print(f"\n[INFO] Sentence 1 fields: {dir(s1) if hasattr(s1, '__dict__') else 'N/A'}")
#     print(f"[INFO] Sentence content: {getattr(s1, 'content', 'N/A')}")
    
#     # Inspect EEG dimensions
#     raw_eeg = getattr(s1, 'rawData', None)
#     if raw_eeg is not None:
#         print(f"[INFO] EEG rawData shape: {np.asarray(raw_eeg).shape}")
#     else:
#         print("[WARNING] 'rawData' not found in sentenceData")

#     # Inspect word structure
#     words = getattr(s1, 'word', None)
#     if words is not None and len(words) > 0:
#         w1 = words[0]
#         print(f"\n[INFO] Word 1 fields: {dir(w1) if hasattr(w1, '__dict__') else 'N/A'}")
        
#         # Check for key eye-tracking fields
#         eye_fields = ['GD', 'TRT', 'FFD', 'GPT', 'meanPupilSize', 'nFixations']
#         for f in eye_fields:
#             val = getattr(w1, f, 'MISSING')
#             print(f"[INFO] Word 1 - {f}: {val}")
#     else:
#         print("[WARNING] No 'word' structure found or list empty.")

# if __name__ == "__main__":
#     # REPLACE THIS PATH WITH YOUR ACTUAL FILE PATH
#     path = '/mnt/DATA/SPLAB/1. Code/EEG-To-Text/dataset/ZuCo/q3zws/osfstorage/task2 - NR/Matlab files/resultsZAB_NR.mat'
#     if os.path.exists(path):
#         inspect_file(path)
#     else:
#         # print(f"[ERROR] File not found: {path}")


import scipy.io as sio

path = "/mnt/DATA/SPLAB/1. Code/EEG-To-Text/dataset/ZuCo/q3zws/osfstorage/task2 - NR/Matlab files/resultsZAB_NR.mat"  # change to one of your files

try:
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    sd = mat["sentenceData"]
    print("Loaded with scipy. Sentences:", len(sd))
    s = sd[0]
    print("Sentence fields:", s._fieldnames)
    print("Text:", s.content)
    w = s.word[0]
    print("Word fields:", w._fieldnames)
except NotImplementedError:
    # newer MATLAB v7.3 files need h5py
    import h5py
    f = h5py.File(path, "r")
    print("Top-level keys:", list(f.keys()))
    print("sentenceData keys:", list(f["sentenceData"].keys()))