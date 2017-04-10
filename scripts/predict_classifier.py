import os
import sys
import csv
from collections import OrderedDict

sys.path.append('/data/code/')
from keras.models import load_model
from kdsb17.trainutils import Generator3dCNN
from kdsb17.fileutils import makedir
from kdsb17.layers import SpatialPyramidPooling3D


def predict_classifier(model_name=None, dataset=None):
    """
    Args:
        model_name (None or str): Path to model file relative to models folder.
            e.g. '20170407_231214/weights.03-0.57.hdf5'
        dataset (None or str): Path to dataset folder relative to data folder.
            e.g. 'npz_2mm_ks3_05p/test'
    """

    if (model_name is None) and (dataset is None):
        try:
            _, model_name, dataset = sys.argv
        except ValueError:
            print('This script takes exactly three arguments, the model weights path and the dataset path.')

    data_path = os.path.join('/data/data/features/20170407_174348/weights.17-0.432386.hdf5', dataset)
    model_path = '/data/models/classifier'
    exception_template = 'An exception of type {0} occurred. Arguments:\n{1!r}'
    results_path = '/data/results'

    out_path = makedir(os.path.join(results_path, model_name, dataset))

    print('Using model:', model_name)
    print('Using dataset:', dataset)
    print('Using data_path:', data_path)

    # Initialize CSV file
    csv_file = open(os.path.join(out_path, 'stage1_submission.csv'), 'w')
    writer = csv.DictWriter(csv_file, fieldnames=['id', 'cancer'], dialect=csv.excel)
    writer.writeheader()

    # Load binary classification model
    model = load_model(os.path.join(model_path, model_name),
                       custom_objects={'SpatialPyramidPooling3D': SpatialPyramidPooling3D([1, 2, 4])})

    # Make generator for prediction
    data_gen = Generator3dCNN(data_path=data_path, labels_path=None)
    generator = data_gen.for_prediction(array_type='cae3d_features')

    try:
        for patient_id, x, _ in generator:
            prob = model.predict(x)

            row_dict = OrderedDict({'id': patient_id, 'cancer': prob})
            writer.writerow(row_dict)
            csv_file.flush()

            print('Predicted:', patient_id, prob)

    except Exception as ex:
        message = exception_template.format(type(ex).__name__, ex.args)
        print(message)

    finally:
        csv_file.close()


if __name__ == '__main__':
    sys.exit(predict_classifier())
