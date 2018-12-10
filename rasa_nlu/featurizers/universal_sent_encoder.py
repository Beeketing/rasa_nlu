from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

import tensorflow as tf
import tensorflow_hub as hub

from rasa_nlu.featurizers import Featurizer


class UniversalSentenceEncoderFeaturizer(Featurizer):
    """Appends a universal sentence encoding to the message's text_features."""
    # URL of the TensorFlow Hub Module
    TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
    name = "universal_sentence_encoder_featurizer"
    requires = []
    provides = ["text_features"]

    def __init__(self, component_config):
        super(UniversalSentenceEncoderFeaturizer, self).__init__(component_config)
        sentence_encoder = hub.Module(self.TFHUB_URL)
        # Create a TensorFlow placeholder for the input string
        self.input_string = tf.placeholder(tf.string, shape=[None])
        # Invoke `sentence_encoder` in order to create the encoding tensor
        self.encoding = sentence_encoder(self.input_string)
        self._WORD_SPLIT = re.compile(u"([.,!?\"'-<>:;)(])")

        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(),
                          tf.tables_initializer()])

    def train(self, training_data, config, **kwargs):
        for example in training_data.training_examples:
            self.process(example)

    def process(self, message, **kwargs):
        # Get the sentence encoding by feeding the message text and computing
        # the encoding tensor.
        text = self._split(message.text)
        feature_vector = self.session.run(self.encoding,
                                          {self.input_string: [text]})[0]
        # Concatenate the feature vector with any existing text features
        features = self._combine_with_existing_text_features(message, feature_vector)
        # Set the feature, overwriting any existing `text_features`
        message.set("text_features", features)

    def _split(self, line):
        words = []
        for fragment in line.strip().split():
            for token in re.split(self._WORD_SPLIT, fragment):
                words.append(token)
        return " ".join(words)
