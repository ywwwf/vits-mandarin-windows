""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_space      = ' '
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

_letters_ipa_1 = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘ᵻ'̩"

# For Simplified Chinese
_punctuation_sc = '；：，。！？-“”《》、（）…— '
# The numbers are for Pinyin tones
_numbers = '123450'

# Additional symbols
# The special character symbols are used to avoid
# conflict with the letter used in Pinyin
_others = 'ＢＰ'

_punctuation_en = ';:,.!?¡¿\'"«»()'

# Export all symbols:
symbols_en = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

symbols_en_1 = [_pad] + [_space] + list(_punctuation_en) + list(_letters) + list(_letters_ipa_1)

symbols_cmn = [_pad] + list(_punctuation_sc) + list(_letters) + list(_numbers) + list(_others)

symbols_ce = [_pad] + list(_punctuation_sc) + list(_letters) + list(_numbers) + list(_others) + list(_punctuation_en) + list(_letters_ipa)