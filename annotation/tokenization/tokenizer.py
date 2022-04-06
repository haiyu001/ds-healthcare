from typing import Optional, List, Iterable, Iterator, Dict, Tuple
from annotation.tokenization.normalizer import Normalizer
from annotation.tokenization.preprocessor import Preprocessor
from annotation.tokenization.base_tokenizer import BaseTokenizer
from spacy.tokens import Doc
import json


class MetaTokenizer:
    def __init__(self,
                 base_tokenizer: BaseTokenizer,
                 preprocessor: Optional[Preprocessor] = None,
                 normalizer: Optional[Normalizer] = None,
                 text_fields_in_json: Optional[str] = None,
                 meta_fields_to_keep: Optional[str] = None,
                 meta_fields_to_drop: Optional[str] = None,
                 attrs: Tuple[str, str] = ("metadata", "source_text")):

        self.base_tokenizer = base_tokenizer
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        self.text_fields_in_json = text_fields_in_json.split(',') if text_fields_in_json else None
        self.meta_fields_to_drop = meta_fields_to_drop.split(',') if meta_fields_to_drop else None
        self.meta_fields_to_keep = meta_fields_to_keep.split(',') if meta_fields_to_keep else None
        self._metadata, self._source_text = attrs
        Doc.set_extension(self._metadata, default={}, force=True)
        Doc.set_extension(self._source_text, default=None, force=True)

    def __call__(self, record: str) -> Doc:
        source_text, metadata = self.read_record(record)
        text = source_text
        if self.preprocessor:
            text = self.preprocessor.preprocess(text)
        doc = self.base_tokenizer.tokenize(text)
        if self.normalizer:
            doc = self.normalizer.normalize(doc)
        doc._.set(self._metadata, metadata)
        doc._.set(self._source_text, source_text)
        return doc

    def read_record(self, record: str) -> Tuple[str, Dict]:
        """record could be text str or json str, if record is json str, text_fields_in_json is required"""
        try:
            record = json.loads(record)
        except ValueError as e:
            return record, {}
        else:
            if not self.text_fields_in_json:
                raise Exception("Need to specify text field for the source json input.")

            common_text_fields = list(set(record.keys()) & set(self.text_fields_in_json))
            if len(common_text_fields) == 0:
                raise Exception(f"None of text fields {common_text_fields} exit in the source input.")
            elif len(common_text_fields) > 1:
                raise Exception(f"More than one text fields {common_text_fields} exit in the source input.")
            else:
                text = record.pop(common_text_fields[0])

            if self.meta_fields_to_drop and self.meta_fields_to_keep:
                raise Exception(f"Either drop some fields or keep some fields. Cannot do both.")
            elif self.meta_fields_to_drop or self.meta_fields_to_keep:
                drop_fields = self.meta_fields_to_drop or [i for i in record.keys() if i not in self.meta_fields_to_keep]
                for i in drop_fields:
                    if i in record.keys():
                        del record[i]
            return text, record

    def pipe(self, records: Iterable[str]) -> Iterator[Doc]:
        for record in records:
            yield self(record)

    def from_bytes(self):
        pass

    def from_disk(self, path: str, **kwargs):
        pass

    def to_bytes(self, *args, **kwargs):
        pass

    def to_disk(self, path: str, **kwargs):
        pass
