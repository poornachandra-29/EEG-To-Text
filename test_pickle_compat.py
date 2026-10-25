import sys
import numpy
import numpy.core
import numpy.core.numeric
import pickle

# Map newer NumPy module paths to the installed NumPy equivalents.
sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.numeric"] = numpy.core.numeric

path = "dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle"

print("NumPy version:", numpy.__version__)

try:
    with open(path, "rb") as f:
        data = pickle.load(f)

    print("SUCCESS: Dataset loaded")
    print("Object type:", type(data))
    if isinstance(data, (list, dict, tuple)):
        print("Object length:", len(data))

except Exception:
    import traceback
    traceback.print_exc()
