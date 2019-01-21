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
        text = message.text
        if "order_code" in message.get("intent").get("name"):
            for entity_mapper in message.get("entities"):
                if entity_mapper["entity"] == "order_code":
                    return
            order_code = ""
            if len(text.split()) == 1:
                order_code = text
            else:
                spacy_nlp = kwargs.get("spacy_nlp", None)
                doc = spacy_nlp(text)
                for token in doc:
                    deps = list(token.lefts)
                    additional_txt = ""
                    # Get external punctuation
                    if len(deps) > 0 and not deps[0].text.isalnum():
                        additional_txt = deps[0].text
                    if token.pos_ in ["NUM", "PROPN"]:
                        order_code = additional_txt + token.text
                        break
                    if token.head.pos_ == "VERB" and token.dep_ == "attr":
                        order_code = additional_txt + token.text
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
        # Rule-based on single word and is alpha-number
        if len(text.split()) == 1 and not text.isalpha() and not text.isdigit() and text.isalnum() and '@' not in text:
            entity = {
                "entity": "order_code",
                "value": text,
                "start": 1,
                "end": len(text) - 1,
                "confidence": 0.7,
                "extractor": self.name,
            }
            message.set("entities",
                        message.get("entities", []) + [entity],
                        add_to_output=True)
