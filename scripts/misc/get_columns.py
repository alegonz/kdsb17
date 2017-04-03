import os
import time
import datetime
import dicom
import pandas as pd

BASE_PATH = '/data/data/stage1'
OUT_PATH = '/data/results'

suffixes = {
    'ImagePositionPatient': ['x', 'y', 'z'],
    'ImageOrientationPatient': ['Rx', 'Ry', 'Rz', 'Cx', 'Cy', 'Cz'],
    'PatientOrientation': ['R', 'C'],
    'PixelSpacing': ['R', 'C'],
    'PixelAspectRatio': ['R', 'C']
}

window_elems = ['WindowCenter', 'WindowWidth', 'WindowCenterWidthExplanation']

# First quick scan to get all possible column names.
columns = []
n = 0
t0 = time.time()
for p, patient_folder in enumerate(os.listdir(BASE_PATH), start=1):
    
    dcm_files = os.listdir(os.path.join(BASE_PATH, patient_folder))
    
    print('Scanning patient %s (%d files)...' % (patient_folder, len(dcm_files)))
    
    t1 = time.time()
    for dcm_file in dcm_files:
        
        dcm_path = os.path.join(BASE_PATH, patient_folder, dcm_file)
        dcm = dicom.read_file(dcm_path)
        
        elem_names = dcm.__dir__()
        elem_names = [e for e in elem_names if e[0].isupper() and e != 'PixelData']
        
        # list-like elements
        list_like_elems = [e for e in elem_names if e in suffixes.keys()]
        for e in list_like_elems:
            replacement = [e+suffix for suffix in suffixes[e]]
            
            idx = elem_names.index(e)
            elem_names.remove(e)
            for r in reversed(replacement):
                elem_names.insert(idx, r)
        
        new_elems = [e for e in elem_names if e not in columns]
        
        if len(new_elems) > 0:
            print('New elements found:', new_elems)
            columns += new_elems
        
        n += 1
    
    t2 = time.time()
    
    proc_time = str(datetime.timedelta(seconds=t2-t1))
    total_time = str(datetime.timedelta(seconds=t2-t0))
    print('Done. %d lines so far. Proc. time %s. Total time %s.' % (n, proc_time, total_time))
    
    if p % 100 == 0:
        print('100 patients processed. Saving to temporal csv file.')
        pd.DataFrame(columns=columns).to_csv(os.path.join(OUT_PATH, 'header.csv.temp'))

print('There are %d dcm files and %d different columns.' % (n, len(columns)))
pd.DataFrame(columns=columns).to_csv(os.path.join(OUT_PATH, 'header.csv'), index=False)
