df_correction = pd.read_parquet('df_train_corrected.parquet')


def word_correction(string):

    li_string = string.split()

    for i, word in enumerate(li_string):
        mask = df_correction['word'] == word
        if any(mask):
            corrected_word = df_correction.loc[mask, 'correction'].values[0]
            li_string[i] = corrected_word

    corrected_string = ' '.join(li_string)

    return corrected_string
