import sys
import os
import inspect


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
sys.path.append(os.path.join(current_dir, ".."))
