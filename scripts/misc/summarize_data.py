import os
import time
import datetime
import dicom
import pandas as pd

BASE_PATH = '/data/data/stage1'
OUT_PATH = '/data/results'
HEADER_PATH = '/data/results/header.csv'

suffixes = {
    'ImagePositionPatient': ['x', 'y', 'z'],
    'ImageOrientationPatient': ['Rx', 'Ry', 'Rz', 'Cx', 'Cy', 'Cz'],
    'PatientOrientation': ['R', 'C'],
    'PixelSpacing': ['R', 'C'],
    'PixelAspectRatio': ['R', 'C']
}

varlen_elems = [
    'SpecificCharacterSet',
    'WindowCenter', 'WindowWidth', 'WindowCenterWidthExplanation',
    'FrameIncrementPointer',
    'Allergies',
    'LossyImageCompressionRatio'
    ]

# read columns from header file
with open(HEADER_PATH, 'r') as f:
    line = f.read()
    columns = line.rstrip().split(',')

columns = ['Folder', 'Filename'] + columns

# Gather data
csv_file = open(os.path.join(OUT_PATH, 'stage1_data_summary.csv'), 'a')
pd.DataFrame(columns=columns).to_csv(csv_file, header=True, index=False)  # write header
csv_file.flush()

n = 0
t0 = time.time()
for p, patient_folder in enumerate(os.listdir(BASE_PATH), start=1):
    
    dcm_files = os.listdir(os.path.join(BASE_PATH, patient_folder))
    
    df = pd.DataFrame(columns=columns)
    
    t1 = time.time()
    for dcm_file in dcm_files:
        
        dcm_path = os.path.join(BASE_PATH, patient_folder, dcm_file)
        dcm = dicom.read_file(dcm_path)
        
        df.loc[n, 'Folder'] = patient_folder
        df.loc[n, 'Filename'] = dcm_file
        
        elem_names = dcm.__dir__()
        elem_names = [e for e in elem_names if e[0].isupper() and e != 'PixelData']
        
        for elem_name in elem_names:
            try:
                if elem_name in suffixes.keys():
                    # list-like elements
                    if len(dcm.get(elem_name)) == 0:
                        continue
                    for i, suffix in enumerate(suffixes[elem_name]):
                        df.loc[n, (elem_name+suffix)] = dcm.get(elem_name)[i]
                
                elif elem_name in varlen_elems:
                    # variable length elements
                    # squeeze these into a string
                    df.loc[n, elem_name] = str(dcm.get(elem_name))
                
                else:
                    # other elements
                    df.loc[n, elem_name] = dcm.get(elem_name)
            
            except:
                print('Something weird at patient %s file %s element %s' % (patient_folder, dcm_file, elem_name))
        
        n += 1
    
    t2 = time.time()
    
    proc_time = str(datetime.timedelta(seconds=t2-t1))
    total_time = str(datetime.timedelta(seconds=t2-t0))
    
    df.to_csv(csv_file, header=False, index=False)
    csv_file.flush()
    
    print('Patient %s done - %d files - %d - %s - %s - %.3f' % (patient_folder, len(dcm_files), n,
                                                                proc_time, total_time, (t2-t0)/n))
    
csv_file.close()
