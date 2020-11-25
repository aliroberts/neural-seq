

from src.encoders.simple_drum import DrumEncoder
from src.encoders.rich_drum.tokens import load_top_toks


class RichDrumEncoder(DrumEncoder):
    """
    Similar to DrumEncoder except that note and duration tokens are lumped together
    resulting in a larger vocabulary.

    ['P-42', 'P-35', 'D-2',...] will become ['P-42_P-35_D-2',...]

    The number of such aggregate tokens will be equal to the top 500 such tokens that appear
    in the dataset (note that this is the result of prior analysis rather than being computed
    by this encoder). Remaining tokens will be included in a notewise fashion (as per the
    first set of tokens above)
    """
    SMALLEST_DIVISION = 8

    def __init__(self):
        self.top_toks = load_top_toks()

    def simple_to_rich(self, enc):
        # Groups of tokens comprising of an action and rest will be lumped together to form
        # aggregate tokens (if they appear in the top tokens)
        rich_enc = []
        toks = []
        for tok in enc:
            if tok[0] == 'P':
                toks.append(tok)
            else:
                toks.sort()
                toks.append(tok)
                joined = ('_').join(toks)
                if joined in self.top_toks:
                    rich_enc.append(joined)
                else:
                    rich_enc += toks
                toks = []
        return rich_enc

    def rich_to_simple(self, enc):
        simple_enc = []
        for tok in enc:
            simple_enc += tok.split('_')
        return simple_enc

    def encode(self, midi):
        simple_enc = super().encode(midi)
        return self.simple_to_rich(simple_enc)

    def decode(self, enc, midi_programs=None, tempo=None):
        return super().decode(self.rich_to_simple(enc), midi_programs=midi_programs, tempo=tempo)

    def process_prediction(self, enc, **kwargs):
        simple_enc = self.rich_to_simple(enc)
        prediction = self.simple_to_rich(
            super().process_prediction(simple_enc, **kwargs))
        print(prediction)
        return prediction

    def duration(self, enc):
        simple_enc = self.rich_to_simple(enc)
        return super().duration(simple_enc)
