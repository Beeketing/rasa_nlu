from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message


class OrderCodeExtractor(EntityExtractor):
    name = "ner_order_code"

    provides = ["entities"]

    requires = ["spacy_nlp"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER

        # extracted = self.add_extractor_name(self.extract_entities(doc))
        # message.set("entities",
        #             message.get("entities", []) + extracted,
        #             add_to_output=True)
        if message.get("intent") == "order_code":
            for entity_mapper in message.get("entities"):
                if entity_mapper["entity"] != "order_code":
                    return
        text = message.text
        order_code = ""
        if len(text.split()) == 1:
            order_code = text
        else:
            spacy_nlp = kwargs.get("spacy_nlp", None)
            doc = spacy_nlp(text)
            for token in doc:
                if token.pos_ == "NUM":
                    order_code = token.text
                    break
        if order_code != "":
            start_idx = text.find(order_code)
            end_idx = start_idx + len(order_code)
            entity = {
                "entity": "order_code",
                "value": order_code,
                "start": start_idx,
                "end": end_idx,
                "confidence": 1.0,
                "extractor": self.name,
            }
            message.set("entities",
                        message.get("entities", []) + [entity],
                        add_to_output=True)
