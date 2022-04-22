from typing import Optional, Iterable, Iterator, Dict, Tuple
from annotation.tokenization.normalizer import Normalizer
from annotation.tokenization.preprocessor import Preprocessor
from annotation.tokenization.base_tokenizer import BaseTokenizer
from spacy.tokens import Doc
import json


class MetadataTokenizer(object):
    def __init__(self,
                 base_tokenizer: BaseTokenizer,
                 preprocessor: Optional[Preprocessor] = None,
                 normalizer: Optional[Normalizer] = None,
                 text_fields_in_json: Optional[str] = None,
                 metadata_fields_to_keep: Optional[str] = None,
                 metadata_fields_to_drop: Optional[str] = None,
                 ignore_metadata: bool = False,
                 attrs: Tuple[str, str, str] = ("metadata", "source_text", "preprocessed_text")):

        self.base_tokenizer = base_tokenizer
        self.preprocessor = preprocessor
        self.normalizer = normalizer
        self.text_fields_in_json = text_fields_in_json.split(",") if text_fields_in_json else None
        self.meta_fields_to_drop = metadata_fields_to_drop.split(",") if metadata_fields_to_drop else None
        self.meta_fields_to_keep = metadata_fields_to_keep.split(",") if metadata_fields_to_keep else None
        self.ignore_metadata = ignore_metadata
        self._metadata, self._source_text, self._preprocessed_text = attrs
        Doc.set_extension(self._metadata, default={}, force=True)
        Doc.set_extension(self._source_text, default=None, force=True)
        Doc.set_extension(self._preprocessed_text, default=None, force=True)

    def __call__(self, record: str) -> Doc:
        source_text, metadata = self.read_record(record)
        text = source_text

        preprocessed_text = None
        if self.preprocessor:
            preprocessed_text = self.preprocessor.preprocess(text)
            text = preprocessed_text

        doc = self.base_tokenizer.tokenize(text)
        if self.normalizer:
            doc = self.normalizer.normalize(doc)
        doc._.set(self._metadata, metadata)
        doc._.set(self._source_text, source_text)
        doc._.set(self._preprocessed_text, preprocessed_text)
        return doc

    def read_record(self, record: str) -> Tuple[str, Dict]:
        """record could be text str or json str, if record is json str, text_fields_in_json is required"""
        try:
            record = json.loads(record)
        except ValueError:
            return record, {}
        else:
            if not self.text_fields_in_json:
                raise ValueError("Need to specify text field for the source json input.")

            common_text_fields = list(set(record.keys()) & set(self.text_fields_in_json))
            if len(common_text_fields) == 0:
                raise ValueError(f"None of text fields {self.text_fields_in_json} exist in the source input.")
            elif len(common_text_fields) > 1:
                raise ValueError(f"More than one text fields {common_text_fields} exist in the source input.")
            else:
                text = record.pop(common_text_fields[0])

            if self.ignore_metadata:
                return text, {}

            metadata = record
            if self.meta_fields_to_drop and self.meta_fields_to_keep:
                raise ValueError(f"Either drop some fields or keep some fields. Cannot do both.")
            elif self.meta_fields_to_drop or self.meta_fields_to_keep:
                drop_fields = self.meta_fields_to_drop or [i for i in metadata.keys() if
                                                           i not in self.meta_fields_to_keep]
                for i in drop_fields:
                    if i in metadata.keys():
                        del metadata[i]
            return text, metadata

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
