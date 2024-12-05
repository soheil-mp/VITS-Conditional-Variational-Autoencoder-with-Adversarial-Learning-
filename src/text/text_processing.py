import re
from unidecode import unidecode
import inflect

# Text cleaners
class TextCleaner:
    def __init__(self):
        self.word_to_num = inflect.engine()
        self._whitespace_re = re.compile(r'\s+')
        self._acronym_re = re.compile(r'([A-Z][A-Z]+)(?=[^A-Z]|\Z)')
        self._number_re = re.compile(r'([0-9]+)')
        
        # Basic English phoneme set
        self._phonemes = {
            'a': 'AA',  # as in "father"
            'e': 'IY',  # as in "eat"
            'i': 'IH',  # as in "it"
            'o': 'OW',  # as in "go"
            'u': 'UW',  # as in "blue"
            'b': 'B',
            'c': 'K',
            'd': 'D',
            'f': 'F',
            'g': 'G',
            'h': 'HH',
            'j': 'JH',
            'k': 'K',
            'l': 'L',
            'm': 'M',
            'n': 'N',
            'p': 'P',
            'q': 'K',
            'r': 'R',
            's': 'S',
            't': 'T',
            'v': 'V',
            'w': 'W',
            'x': 'K S',
            'y': 'Y',
            'z': 'Z',
            'th': 'TH',
            'ch': 'CH',
            'sh': 'SH',
            'ph': 'F',
            'ng': 'NG',
            ' ': ' ',
            ',': ',',
            '.': '.',
            '!': '!',
            '?': '?',
            "'": '',
            '-': ' ',
            '\n': ' ',
        }
        
        # Special cases
        self._special_cases = {
            'the': 'DH AH',
            'a': 'AH',
            'an': 'AE N',
            'and': 'AE N D',
            'or': 'AO R',
            'for': 'F AO R',
            'to': 'T UW',
            'in': 'IH N',
            'is': 'IH Z',
            'was': 'W AA Z',
            'you': 'Y UW',
            'are': 'AA R',
        }
    
    def normalize_numbers(self, text):
        """Convert numbers to their spoken form."""
        def _replace_number(match):
            number = match.group(0)
            return self.word_to_num.number_to_words(number)
        
        return re.sub(self._number_re, _replace_number, text)
    
    def expand_acronyms(self, text):
        """Convert acronyms to their spoken form."""
        def _expand_acronym(match):
            acronym = match.group(0)
            return ' '.join(list(acronym.lower()))
        
        return re.sub(self._acronym_re, _expand_acronym, text)
    
    def text_to_sequence(self, text, cleaner_names):
        """Convert text to sequence of phoneme IDs."""
        # Clean text
        text = text.lower().strip()
        text = unidecode(text)  # Convert accented characters to ASCII
        text = self.normalize_numbers(text)
        text = self.expand_acronyms(text)
        
        # Split into words
        words = text.split()
        
        # Convert to phonemes
        phonemes = []
        for word in words:
            if word in self._special_cases:
                phonemes.extend(self._special_cases[word].split())
            else:
                # Simple grapheme-to-phoneme conversion
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i:i+2] in self._phonemes:
                        phonemes.append(self._phonemes[word[i:i+2]])
                        i += 2
                    else:
                        if word[i] in self._phonemes:
                            phonemes.append(self._phonemes[word[i]])
                        i += 1
        
        # Convert phonemes to IDs
        sequence = []
        for phoneme in phonemes:
            if phoneme in symbols_to_id:
                sequence.append(symbols_to_id[phoneme])
        
        return sequence

# Create symbol mappings
symbols = ['_PAD', '_EOS', '_BOS', '_UNK']  # Special tokens
symbols.extend(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
               'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
               'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'])
symbols.extend([c for c in '.,!?-~'])  # Basic punctuation

symbols_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbols = {i: s for i, s in enumerate(symbols)}

# Create cleaner instance
_cleaner = TextCleaner()

def text_to_sequence(text, cleaner_names):
    """Convert text to sequence of IDs."""
    return _cleaner.text_to_sequence(text, cleaner_names) 