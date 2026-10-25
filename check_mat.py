import scipy.io as sio
path = (
    "dataset/ZuCo/q3zws/osfstorage/"
    "task1- SR/Matlab files/resultsZDM_SR.mat"
)

print("Loading sentenceData...", flush=True)

try:
    data = sio.loadmat(
        path,
        variable_names=["sentenceData"],
        squeeze_me=True,
        struct_as_record=False
    )

    print("Loaded successfully!", flush=True)
    print("Shape:", data["sentenceData"].shape, flush=True)

except Exception as e:
    print("Error type:", type(e).__name__, flush=True)
    print("Error:", str(e), flush=True)
