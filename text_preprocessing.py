from emoji import demojize
from unicodedata import category, name, normalize
import re
import string

spaces = [
    '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061',
    '\x10', '\x7f', '\x9d', '\xad', '\xa0'
]


def remove_space(text):

    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)

    return text


def remove_diacritics(s):
    return ''.join(c for c in normalize(
        'NFKD',
        s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace(
            '₋', '-')) if category(c) != 'Mn')


special_punc_mappings = {
    "—": "-",
    "–": "-",
    "_": "-",
    '”': '"',
    "″": '"',
    '“': '"',
    '•': '.',
    '−': '-',
    "’": "'",
    "‘": "'",
    "´": "'",
    "`": "'",
    '\u200b': ' ',
    '\xa0': ' ',
    '،': '',
    '„': '',
    '…': ' ... ',
    '\ufeff': ''
}


def clean_special_punctuations(text):

    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    text = remove_diacritics(text)

    return text


def clean_number(text):

    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'\d+', '', text)

    return text


regular_punct = list(string.punctuation)
extra_punct = [
    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&', '/', '[',
    ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{',
    '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§',
    '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−',
    '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓',
    '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄',
    '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙',
    '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔',
    '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡',
    '§', '£', '₤'
]
all_punct = list(set(regular_punct + extra_punct))


def remove_punctuation(text):

    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
            text = text.replace(punc, '')

    return text


def remove_elongation(input_string):

    pattern = re.compile(r'(.)\1{2,}')
    result = re.sub(pattern, r'\1', input_string)

    return result


def convert_emoji_to_text(emoji_text):
    text_with_aliases = demojize(emoji_text)
    return text_with_aliases


def preprocess(text):

    text = text.lower()
    text = convert_emoji_to_text(text)
    text = remove_space(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = remove_punctuation(text)
    text = remove_elongation(text)
    text = remove_space(text)

    return text
