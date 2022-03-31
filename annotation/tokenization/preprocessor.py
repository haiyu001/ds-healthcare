from textacy.preprocessing.remove import accents as remove_accents
from textacy.preprocessing.replace import emails as replace_emails
from textacy.preprocessing.replace import urls as replace_urls
from textacy.preprocessing.replace import hashtags as replace_hashtags
from textacy.preprocessing.replace import user_handles as replace_user_handles
from textacy.preprocessing.normalize import repeating_chars as normalize_repeating_chars
from textacy.preprocessing.normalize import whitespace as normalize_whitespace
from textacy.preprocessing.resources import RE_LINEBREAK
import re


class Preprocessor:

    def __init__(self,
                 to_lowercase: bool = False,
                 norm_whitespace: bool = True,
                 norm_repeating_punctuation: bool = True,
                 rm_accents: bool = True,
                 rm_linebreak: bool = True,
                 rm_emails: bool = False,
                 rm_urls: bool = False,
                 rm_hashtags: bool = False,
                 rm_handles: bool = False,
                 rp_emails: bool = False,
                 rp_urls: bool = False,
                 rp_hashtags: bool = False,
                 rp_handles: bool = False):

        self.to_lowercase = to_lowercase
        self.norm_whitespace = norm_whitespace
        self.norm_repeating_punctuation = norm_repeating_punctuation
        self.rm_accents = rm_accents
        self.rm_linebreak = rm_linebreak
        self.rm_emails = rm_emails
        self.rm_urls = rm_urls
        self.rm_hashtags = rm_hashtags
        self.rm_handles = rm_handles
        self.rp_emails = rp_emails
        self.rp_urls = rp_urls
        self.rp_hashtags = rp_hashtags
        self.rp_handles = rp_handles

    def preprocess(self, text: str) -> str:
        if self.rm_accents:
            text = remove_accents(text)

        if self.rm_linebreak:
            text = RE_LINEBREAK.sub(r" ", text)

        if self.rp_emails or self.rm_emails:
            text = replace_emails(text, repl="RP_EMAIL" if not self.rm_emails else " ")

        if self.rp_urls or self.rm_urls:
            text = replace_urls(text, repl="RP_URL" if not self.rm_urls else " ")

        if self.rp_hashtags or self.rm_hashtags:
            text = replace_hashtags(text, repl="RP_HASHTAG" if not self.rm_hashtags else " ")

        if self.rp_handles or self.rm_handles:
            text = replace_user_handles(text, repl="RP_HANDLE" if not self.rm_handles else " ")

        if self.norm_repeating_punctuation:
            for p in ",;:":
                if p == "\\":
                    text = re.sub(r"(\\){2,}", "\\\\", text)
                else:
                    text = normalize_repeating_chars(text, chars=p, maxn=1)

        if self.norm_whitespace:
            text = normalize_whitespace(text)

        if self.to_lowercase:
            text = text.lower()

        return text
