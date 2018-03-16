import os
# Force backend to be theano (vs tensorflow) for compatibility with model
os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing import sequence
from keras.models import load_model


class DGAClassifier(object):
    def __init__(self, model_fn):
        self._model = load_model(model_fn)

        # maxlen of domain (used for padding) as defined during train time
        # Should not be changed
        self._maxlen = 100

    def predict_on_file(self, fn):
        """
        Predict on a file of domain names whether domain is DGA or not

        :param fn: filename of txt file of domain names
        :type fn: str
        :return: array of probabilities that domains are DGA
        :rtype: numpy.ndarray
        """
        data = []
        with open(fn, 'r') as fin:
            for line in fin:
                data.append(line.strip())

        data = [[ord(c) for c in domain] for domain in data]
        data_feat = sequence.pad_sequences(data,
                                           maxlen=self._maxlen,
                                           value=-1)
        results = self._model.predict(data_feat)
        return results

    def predict(self, domains):
        """
        Predict on a list of domain names whether domain is DGA or not

        :param domains: list of domain names
        :type fn: list
        :return: array of probabilities that domains are DGA
        :rtype: numpy.ndarray
        """
        data = [[ord(c) for c in d] for d in domains]
        data_feat = sequence.pad_sequences(data,
                                           maxlen=self._maxlen,
                                           value=-1)
        results = self._model.predict(data_feat)
        return results


if __name__ == '__main__':
    # First create classifier. This can take a while
    classifier = DGAClassifier('dga-bot.h5')

    # But subsequent inferences are fast. Predict on list of domain names
    print(classifier.predict(['google.com',
                              'asdhcvuagq.com',
                              'correcthorsebatterystaple.com']))

    # Or pass in text files containing domain names
    print(classifier.predict_on_file('sample_dga.txt'))
    print(classifier.predict_on_file('sample_legit.txt'))
