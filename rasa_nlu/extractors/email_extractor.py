from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from typing import Any

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message


class EmailExtractor(EntityExtractor):
    name = "ner_email"

    provides = ["entities"]

    email_regex = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        text = message.text
        entities = []
        for token in text.split():
            matcher = self.email_regex.match(token)
            if matcher:
                entities.append({
                    "entity": "email",
                    "value": token,
                    "start": matcher.start(),
                    "end": matcher.end(),
                    "confidence": 1.0,
                    "extractor": self.name
                })
            message.set("entities",
                        message.get("entities", []) + entities,
                        add_to_output=True)
