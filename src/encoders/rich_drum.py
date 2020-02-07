
from .simple_drum import DrumEncoder


class RichDrumEncoder(DrumEncoder):
    """
    Similar to DrumEncoder except that note and duration tokens are lumped together
    resulting in a larger vocabulary.

    ['P-42', 'P-35', 'D-2',...] will become ['P-42_P-35_D-2',...]

    The number of such aggregate tokens will be equal to the top 500 such tokens that appear
    in the dataset (note that this is the result of prior analysis rather than being computed
    by this encoder). Remaining tokens will be included in a notewise fashion (as per the
    first set of tokens above).
    """

    @staticmethod
    def rich_to_simple(enc):
        simple_enc = []
        for tok in enc:
            simple_enc += tok.split('_')
        return simple_enc

    @staticmethod
    def simple_to_rich(enc):
        # Groups of tokens comprising of an action and rest will be lumped together to form
        rich_enc = []
        toks = []
        for tok in enc:
            if tok[0] == 'D':
                toks.append(tok)
                rich_enc.append('_'.join(toks))
                toks = []
            else:
                toks.append(tok)
        return rich_enc

    def encode(self, midi):
        simple_enc = super().encode(midi)
        return self.simple_to_rich(simple_enc)

    def decode(self, enc, midi_programs=None, tempo=None):
        return super().decode(self.rich_to_simple(enc), midi_programs=None, tempo=None)

    def process_prediction(self, enc):
        simple_enc = self.rich_to_simple(enc)
        return self.simple_to_rich(super().process_prediction(simple_enc))

    def duration(self, enc):
        simple_enc = self.rich_to_simple(enc)
        return self.simple_to_rich(super().duration(simple_enc))
