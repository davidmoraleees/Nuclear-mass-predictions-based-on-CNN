import pandas as pd

df = pd.read_fwf(
    'AME2016.txt',
    usecols=(1, 2, 3, 4, 6, 9, 10, 11, 13, 16, 17, 21, 22),
    names=['N-Z', 'N', 'Z', 'A', 'Element', 'mass_exc',
           'mass_exc_unc', 'bind_ene', 'bind_ene_unc', 'beta_ene',
           'beta_ene_unc', 'atomic_mass', 'atomic_mass_unc'],
    widths=(1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 11, 1, 10, 1, 2, 13, 15, 4, 3, 2, 18, 12),
    header=31,
    index_col=False
)

print(df.head())
