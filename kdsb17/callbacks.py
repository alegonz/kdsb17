import csv

from collections import OrderedDict

from keras.callbacks import Callback


class BatchLossCSVLogger(Callback):
    """Callback that streams the batch loss to a csv file.
    # Example
        ```python
            csv_logger = BatchLossCSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.csv_file = None
        self.append = append
        self.writer = None
        self.append_header = True
        self.epoch = 0
        super(BatchLossCSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'batch', 'loss'], dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': self.epoch, 'batch': batch, 'loss': logs.get('loss')})
        # row_dict.update({'loss': logs.get('loss')})
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
