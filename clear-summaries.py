import os
import shutil

file_path = os.path.dirname(os.path.realpath(__file__))

for f in os.listdir(os.path.join(file_path, 'summaries')):
    if f != '.gitkeep':
        shutil.rmtree(os.path.join(file_path, 'summaries', f))

print('Done')
