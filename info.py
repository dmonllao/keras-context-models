import sys
import os
import time
import webbrowser

import numpy as np
import pandas as pd
import pandas_profiling

if len(sys.argv[1]) < 1:
    print('Error: No file path provided')
    exit(1)

skiprows = 2
filepath = sys.argv[1]

filename = str(time.time()) + '-' + os.path.basename(filepath)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filepath = os.path.join(script_dir, 'garbage', filename + '.html')

df = pd.read_csv(filepath, parse_dates=True, encoding='UTF-8',
                 skiprows=skiprows, dtype=np.float32)
profile = pandas_profiling.ProfileReport(df, correlation_threshold=0.7)

profile.to_file(outputfile=output_filepath)

webbrowser.open_new_tab(output_filepath)
