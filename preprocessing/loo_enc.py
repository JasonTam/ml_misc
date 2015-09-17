import numpy as np

def loo_mean_enc(df, col, col_target, col_split='split',
            split_train_flag='Train', split_test_flag='Test',
            mult_noise=np.random.normal):
    """ Leave-one-out mean encoding of categorical features as described by Owen Zhang in his slideshow
    http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions

    :param df: dataframe containing the categorical column and target column
    :param col: name of categorical column to be encoded
    :param col_target: name of target column to be used for means
    :param col_split: name of column that distinguishes train from test set
    :param split_train_flag: flag value in `col_split` that denotes training row
    :param split_test_flag: flag value in `col_split` that denotes testing row
    :param mult_noise: multiplicative noise generator. If `None`, no noise is multiplied
    :return:
    """

    if mult_noise is None:
        mult_noise = lambda size: 1

    df_tmp = pd.DataFrame(index=df.index)
    col_enc = col + ' enc'
    df_tmp[col_enc] = None  # Make empty column

    g_split_col = df.groupby([col_split, col], axis=0)
    g_col = df.groupby([col_split], axis=0)

    # Full means (to fill test set) (apply bool-index-wise)
    gm = g_split_col.mean()

    df_tmp[col_enc][g_col.groups[split_test_flag]] = [
        gm.loc[split_train_flag, col_val].values[0]
        for col_val in df[col][g_col.groups[split_test_flag]]]

    # LOO for training (apply row-wise)
    sum_len_map = {
        val: np.array([
            df[col_target][(df[col] == val) & (df[col_split] == split_train_flag)].sum(axis=0),
            sum((df[col] == val) & (df[col_split] == split_train_flag))
        ]) for val in df[col].unique()}

    def loo_mean(row):
        return np.divide(*sum_len_map[row[col]] - [row[col_target], 1])

    df_tmp[col_enc][df[col_split] == split_train_flag] = df.loc[df[col_split] == split_train_flag].apply(loo_mean, axis=1)

    # Random noise
    df_tmp['random'] = np.random.normal(loc=1.0, scale=0.01, size=len(df))
    df_tmp['random'][df[col_split] == split_test_flag] = 1

    df_tmp[col_enc + ' noisy'] = df_tmp[col_enc] * df_tmp['random']
    return df_tmp[col_enc + ' noisy']


if __name__ == '__main__':
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'

    # Make dummy dataset
    df = pd.DataFrame()
    df['split'] = ['Train'] * 4 + ['Test'] * 3 + ['Train']
    df['User ID'] = ['A1'] * 6 + ['A2'] * 2
    df['Y'] = [0, 1, 1, 0, None, None, None, 0]

    # Encoding process
    col = 'User ID'
    col_enc = 'User ID enc'
    col_target = 'Y'

    df['encoded'] = loo_mean_enc(df, col, col_target)

