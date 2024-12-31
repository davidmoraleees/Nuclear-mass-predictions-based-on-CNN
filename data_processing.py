import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.colors import TwoSlopeNorm
from utils import fontsizes, plot_data

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

uma = config['LDM']['uma']
m_n =  config['LDM']['m_n']*(10**-6)*uma 
m_H =  config['LDM']['m_H']*(10**-6)*uma 
m_e = config['LDM']['m_e']
av = config['LDM']['av']
aS = config['LDM']['aS']
ac = config['LDM']['ac']
aA = config['LDM']['aA']
ap = config['LDM']['ap']
remove_hashtags = config['data']['remove_hashtags']
data_folder = 'data'
data_processing_plots = 'Data processing plots' #Plots folder of AME2016 dataset

fontsizes(config)

# Extracting data from WS4 file
with open(f'{data_folder}/WS4.txt', 'r') as file:
    for i in range(29):
        if i == 27:  
            header = file.readline().strip().split() 
        else:
            next(file) 
    data = [line.strip() for line in file]

data_rows = [line.split() for line in data]
dfWS4= pd.DataFrame(data_rows, columns=header)
dfWS4['N'] = dfWS4['A,'].astype(int) - dfWS4['Z,'].astype(int)
dfWS4 = dfWS4[(dfWS4['N'] >= 8) & (dfWS4['N'] < 180) & (dfWS4['Z,'].astype(int) >= 8) & (dfWS4['Z,'].astype(int) < 120)]
dfWS4['Mth_MeV'] = dfWS4['Mth']
dfWS4['Mth'] = dfWS4['Mth'].astype(float)/uma # u
dfWS4['atomic_mass_ws4'] = dfWS4['Mth'] + dfWS4['A,'].astype(int) # u
dfWS4['atomic_mass_ws4'] = dfWS4['atomic_mass_ws4']*uma # MeV
dfWS4['bind_ene_total_ws4'] = (dfWS4['Z,'].astype(int)*m_H + dfWS4['N']*m_n) - dfWS4['atomic_mass_ws4'] # MeV
dfWS4['Z'] = dfWS4['Z,']
dfWS4['A'] = dfWS4['A,']
dfWS4[['A', 'Z', 'N']] = dfWS4[['A', 'Z', 'N']].astype(int)
dfWS4 = dfWS4.drop(columns=['A,', 'Z,'])
dfWS4['B_e'] = (14.4381*(dfWS4['Z']**2.39) + 1.55468*(10**-6)*(dfWS4['Z']**5.35)) * (10**-6) # MeV
dfWS4['M_N_ws4'] = dfWS4['atomic_mass_ws4'] - dfWS4['Z']*m_e + dfWS4['B_e'] # MeV
dfWS4.to_csv(f'{data_folder}/WS4_cleaned.csv', index=False, header=True, sep=';')


def filter_rows_with_hashtags(df, remove):
    if remove:
        mask = ~df.apply(lambda row: row.astype(str).str.contains('#').any(), axis=1)
        df_filtered = df[mask]
    else:
        df_filtered = df.replace({'#': ''}, regex=True)
    return df_filtered


def process_file(filename, header, widths, columns, column_names, year, remove):
    '''Function to extract data from AME2020 and AME2016 files'''
    df = pd.read_fwf(filename, usecols=columns, names=column_names, widths=widths, header=header, index_col=False)
    df = filter_rows_with_hashtags(df, remove)
    df = df[(df['N'] >= 8) & (df['N'] < 180) & (df['Z'] >= 8) & (df['Z'] < 120)]  # Restrictions for our study case

    df[['Z', 'A']] = df[['Z', 'A']].astype(int)
    df['delta'] = np.where((df['Z'] % 2 == 0) & (df['N'] % 2 == 0), -1, # Z and N even 
                  np.where(df['A'] % 2 == 1, 0, # A odd
                  np.where((df['Z'] % 2 == 1) & (df['N'] % 2 == 1), 1, np.nan))) # Z and N odds
    df['delta'] = df['delta'].astype(int)
    df['delta_I4'] = ((-1)**df['N'] + (-1)**df['Z']) // 2

    A, Z, N, delta = df['A'], df['Z'], df['N'], df['delta']

    df['bind_ene'] = df['bind_ene'].astype(float) / 1000  # MeV
    df['bind_ene_total'] = df['bind_ene'] * A  # MeV
    df['bind_ene_teo'] = (av*A - aS*A**(2/3) - ac*(Z**2)*(A**(-1/3)) - aA*((A-2*Z)**2)/A - ap*delta*(A**(-1/2))) / A  # MeV
    df['bind_ene_teo_total'] = df['bind_ene_teo'] * A  # MeV
    df['Diff_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']  # MeV
    df['Diff_bind_ene_total'] = df['bind_ene_total'] - df['bind_ene_teo_total']  # MeV
    
    df['A2'] = df['A2'].astype(str)
    df['atomic_mass'] = pd.to_numeric(df['A2'] + df['atomic_mass'], errors='coerce') * (10**-6) * uma  # MeV
    df['atomic_mass_unc'] = pd.to_numeric(df['atomic_mass_unc'], errors='coerce') * (10**-6) * uma  # MeV
    df['atomic_mass_teo'] = Z*m_H + N*m_n - df['bind_ene_teo_total']  # MeV
    df['atomic_mass_calc'] = Z*m_H + N*m_n - df['bind_ene_total']  # MeV
    df['Diff_atomic_mass'] = df['atomic_mass'] - df['atomic_mass_calc']  # MeV
    
    df['mass_exc'] = df['mass_exc'].astype(float) # keV
    df['mass_excess_calc'] = (df['atomic_mass']/uma - A) * uma * 1000  # keV
    df['Diff_mass_excess'] = df['mass_exc'] - df['mass_excess_calc']  # keV

    df['bind_ene_calc'] = ((Z*m_H + N*m_n) - df['atomic_mass'])/A  # MeV
    df['Diff_bind_ene_calcs'] = df['bind_ene'] - df['bind_ene_calc'] # MeV
    
    df['B_e'] = (14.4381*(Z**2.39) + 1.55468*(10**-6)*(Z**5.35)) * (10**-6)  # MeV
    df['M_N_teo'] = df['atomic_mass_teo'] - Z*m_e + df['B_e']  # MeV
    df['M_N_exp'] = df['atomic_mass'] - Z*m_e + df['B_e']  # MeV
    df['Diff_nuclear_mass'] = df['M_N_exp'] - df['M_N_teo']  # MeV
    
    return df

columns_2020 = (1, 2, 3, 4, 6, 9, 10, 11, 13, 16, 17, 19, 21, 22)
widths_2020 = (1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 13, 1, 10, 1, 2, 13, 11, 1, 3, 1, 13, 12, 1)
column_names_2020 = ['N-Z', 'N', 'Z', 'A', 'Element', 'mass_exc', 'mass_exc_unc', 'bind_ene', 'bind_ene_unc',
                     'beta_ene', 'beta_ene_unc', 'A2', 'atomic_mass', 'atomic_mass_unc']
header_2020 = 28

columns_2016 = (1, 2, 3, 4, 6, 9, 10, 11, 12, 15, 16, 18, 20, 21)
widths_2016 = [1, 3, 5, 5, 5, 1, 3, 4, 1, 13, 11, 11, 9, 1, 2, 11, 9, 1, 3, 1, 12, 11, 5]
column_names_2016 = ['index', 'N-Z', 'N', 'Z', 'A', 'empty', 'Element', 'empty2', 'empty3', 'mass_exc', 
                     'mass_exc_unc', 'bind_ene', 'bind_ene_unc', 'empty4', 'empty5', 'beta_ene', 'beta_ene_unc',
                     'empty6', 'A2', 'empty7', 'atomic_mass', 'atomic_mass_unc']
header_2016 = 31

df2020 = process_file(f'{data_folder}/mass2020.txt', header_2020, widths_2020, columns_2020, column_names_2020, 2020, remove_hashtags)
df2016 = process_file(f'{data_folder}/mass2016.txt', header_2016, widths_2016, columns_2016, column_names_2016, 2016, remove_hashtags)

if remove_hashtags:
    df2016.to_csv(f'{data_folder}/mass{2016}_cleaned_without_#.csv', sep=';', index=False)
    df2020.to_csv(f'{data_folder}/mass{2020}_cleaned_without_#.csv', sep=';', index=False)
else:
    df2016.to_csv(f'{data_folder}/mass{2016}_cleaned_with_#.csv', sep=';', index=False)
    df2020.to_csv(f'{data_folder}/mass{2020}_cleaned_with_#.csv', sep=';', index=False)

# Merging M_N_ws4 to AME2016 and AME2020 datasets
file_2016 = f"{data_folder}/mass2016_cleaned_with_#.csv"
file_2020 = f"{data_folder}/mass2020_cleaned_with_#.csv"

df2016 = pd.read_csv(file_2016, sep=";")
df2020 = pd.read_csv(file_2020, sep=";")

df2016 = df2016.merge(dfWS4[['Z', 'N', 'M_N_ws4']], on=['Z', 'N'], how='left')
df2020 = df2020.merge(dfWS4[['Z', 'N', 'M_N_ws4']], on=['Z', 'N'], how='left')

df2016['WS4_diff'] = df2016['M_N_exp'] - df2016['M_N_ws4']
df2020['WS4_diff'] = df2020['M_N_exp'] - df2020['M_N_ws4']

df2016.to_csv(file_2016, sep=";", index=False)
df2020.to_csv(file_2020, sep=";", index=False)


def calculate_rmse(df, metrics):
    for metric in metrics:
        rmse = np.sqrt(np.mean(df[metric['column']] ** 2))
        print(f"{metric['label']}: {rmse} {metric['unit']}")

metrics = [
    {'column': 'Diff_bind_ene', 'label': 'RMSE liquid droplet model (2016) binding energy per nucleon', 'unit': 'MeV'},
    {'column': 'Diff_bind_ene_total', 'label': 'RMSE liquid droplet model (2016) binding energy', 'unit': 'MeV'},
    {'column': 'Diff_atomic_mass', 'label': 'RMSE between atomic masses in AME2016 and calculated ones', 'unit': 'MeV'},
    {'column': 'Diff_mass_excess', 'label': 'RMSE between mass excess from AME and calculated ones', 'unit': 'keV'},
    {'column': 'Diff_bind_ene_calcs', 'label': 'RMSE between binding energies per nucleon in AME2016 and calculated ones', 'unit': 'MeV'},
    {'column': 'Diff_nuclear_mass', 'label': 'RMSE liquid droplet model (2016) nuclear masses', 'unit': 'MeV'},
    {'column': 'WS4_diff', 'label': 'RMSE between nuclear masses in AME2016 and WS4', 'unit': 'MeV'}]

calculate_rmse(df2016, metrics)

#Plot of the theoretical binding energy per nucleon as a function of Z and N
plot_data(df2016, 'bind_ene_teo', '(MeV)', 'bind_teo_per_nucleon.pdf',
          data_processing_plots, cmap='jet', vmin=df2016['bind_ene_teo'].min(), vmax=df2016['bind_ene_teo'].max())

#Plot of the experimental binding energy per nucleon as a function of Z and N
plot_data(df2016, 'bind_ene', '(MeV)', 'bind_exp_per_nucleon.pdf',
          data_processing_plots, cmap='jet', vmin=df2016['bind_ene_teo'].min(), vmax=df2016['bind_ene_teo'].max())

#Plot of the difference between theoretical and experimental binding energies per nucleon
plot_data(df2016, 'Diff_bind_ene', '(MeV)', 'bind_teoexp_dif_per_nucleon.pdf',
          data_processing_plots, cmap='seismic', vmin=df2016['Diff_bind_ene'].min(), vcenter=0, vmax=df2016['Diff_bind_ene'].max())

#Plot of the difference between mass excess from AME and the one calculated
plot_data(df2016, 'Diff_mass_excess', '(keV)',
          'mass_excess_expcalc_dif.pdf', data_processing_plots, cmap='seismic') 
                                   
#Plot of the difference between atomic mass from AME and the one calculated                                  
plot_data(df2016, 'Diff_atomic_mass', '(MeV)',
          'atomic_mass_expcalc_dif.pdf', data_processing_plots, cmap='seismic')

#Plot of the experimental atomic mass                                        
plot_data(df2016, 'atomic_mass', '(MeV)',
          'atomic_mass_exp.pdf', data_processing_plots, cmap='seismic')

#Plot of the theoretical atomic mass                                        
plot_data(df2016, 'atomic_mass_teo', '(MeV)',
          'atomic_mass_teo.pdf', data_processing_plots, cmap='seismic')

#Plot of the difference between binding energy per nucleon from AME and the one calculated                                      
plot_data(df2016, 'Diff_bind_ene_calcs', '(MeV)',
          'bind_ene_expcalc_dif.pdf', data_processing_plots, cmap='seismic')

#Plot of the experimental nuclear mass from AME as a function of Z and N                                      
plot_data(df2016, 'M_N_exp', '(MeV)',
          'nuclear_mass_exp.pdf', data_processing_plots, cmap='seismic')

#Plot of the theoretical nuclear mass as a function of Z and N                                      
plot_data(df2016, 'M_N_teo', '(MeV)',
          'nuclear_mass_teo.pdf', data_processing_plots, cmap='seismic')

#Plot of the difference between nuclear mass calculated and the one from liquid-drop model                                       
plot_data(df2016, 'Diff_nuclear_mass', r'$\Delta$ (MeV)',
          'nuclear_mass_expteo_dif.pdf', data_processing_plots, cmap='seismic', vmin=-14, vcenter=0, vmax=14)

#Plot of the difference between nuclear mass calculated and the one from WS4 model    
plot_data(df2016, 'WS4_diff', r'$\Delta$ (MeV)',
          'nuclear_mass_expws4_dif.pdf', data_processing_plots, cmap='seismic')


#3D plot of the difference between theoretical and experimental binding energies per nucleon
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  
norm = TwoSlopeNorm(vmin=df2016['Diff_bind_ene'].min(), vcenter=0, vmax=df2016['Diff_bind_ene'].max())
scatter = ax.scatter(df2016['Z'], df2016['N'], df2016['Diff_bind_ene'], c=df2016['Diff_bind_ene'],
                     cmap='seismic', norm=norm, edgecolor='None', s=12)
ax.set_xlabel('Z')
ax.set_ylabel('N')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('(MeV)')
plt.savefig(os.path.join(data_processing_plots, 'bind_teoexp_dif_3D_per_nucleon.pdf'))
plt.close()


#Nuclear shell gaps (\Delta_{2n} and \Delta_{2p})
def calculate_shell_gaps(df, element, axis, type, column, year, remove):
    '''Neutrons--> element = n, axis = Z; Protons--> element = p, axis = N
    type = exp or teo; column = bind_ene_total or bind_ene_teo_total '''
    df[f'bind_ene_{element}+2_{type}'] = df.groupby(axis)[column].shift(-2)
    df[f'bind_ene_{element}-2_{type}'] = df.groupby(axis)[column].shift(2)
    df[f'delta_2{element}_{type}'] = df[f'bind_ene_{element}-2_{type}'] - 2 * df[column] + df[f'bind_ene_{element}+2_{type}']
    if remove:
        df.to_csv(f'{data_folder}/mass{year}_cleaned_without_#.csv', sep=';', index=False)
    else:
        df.to_csv(f'{data_folder}/mass{year}_cleaned_with_#.csv', sep=';', index=False)
    return df


def plot_shell_gaps(df, gap_col, type, filename, data_processing_plots, vmin, vmax, xlim, ylim): #gap_col=delta_2n or delta_2p
    plot_name = "{}_{}".format(gap_col, type)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['N'], df['Z'], c=df[plot_name], cmap='jet', edgecolor='None', s=12, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(scatter)
    cbar.set_label('(MeV)')

    magic_numbers = [8, 20, 28, 50, 82, 126]
    for magic in magic_numbers:
        plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
        plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)

    plt.xticks(magic_numbers)
    plt.yticks(magic_numbers)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('N')
    plt.ylabel('Z')
    plt.savefig(os.path.join(data_processing_plots, filename))
    plt.close()

df2016 = calculate_shell_gaps(df2016, 'n', 'Z', 'exp', 'bind_ene_total', 2016, remove_hashtags) 
df2016 = calculate_shell_gaps(df2016, 'p', 'N', 'exp', 'bind_ene_total', 2016, remove_hashtags) 
df2016 = calculate_shell_gaps(df2016, 'n', 'Z', 'teo', 'bind_ene_teo_total', 2016, remove_hashtags) 
df2016 = calculate_shell_gaps(df2016, 'p', 'N', 'teo', 'bind_ene_teo_total', 2016, remove_hashtags) 

min_value = min(df2016['delta_2n_exp'].min(), df2016['delta_2p_exp'].min(),
                df2016['delta_2n_teo'].min(), df2016['delta_2p_teo'].min()) #Same colorbar range for both plots
max_value = max(df2016['delta_2n_exp'].max(), df2016['delta_2p_exp'].max(),
                df2016['delta_2n_teo'].max(), df2016['delta_2p_teo'].max())

xlim = (min(df2016['N'].min(), 0), max(df2016['N'].max() + 10, 0)) #Same limits for both plots
ylim = (0, df2016['Z'].max() + 10)  

plot_shell_gaps(df2016, 'delta_2n', 'exp', 'neutron_shell_gaps_exp.pdf', data_processing_plots,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2p', 'exp', 'proton_shell_gaps_exp.pdf', data_processing_plots,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2n', 'teo', 'neutron_shell_gaps_teo.pdf', data_processing_plots,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2p', 'teo', 'proton_shell_gaps_teo.pdf', data_processing_plots,
                min_value, max_value, xlim=xlim, ylim=ylim)
                    

def calculate_difference(df1, df2, output_file):
    diff = df1.merge(df2, on=['Z', 'N'], how='left', indicator=True, suffixes=('', '_other'))
    diff = diff[diff['_merge'] == 'left_only'].drop(columns=['_merge'])
    diff = diff.dropna(axis=1, how='all')
    diff.to_csv(output_file, index=False, sep=';')

# yes = contains '#';   no = doesn't contain '#'
df2016_no = process_file(f'{data_folder}/mass2016.txt', header_2016, widths_2016, columns_2016, column_names_2016, 2016, True)
df2016_yes = process_file(f'{data_folder}/mass2016.txt', header_2016, widths_2016, columns_2016, column_names_2016, 2016, False)
df2020_no = process_file(f'{data_folder}/mass2020.txt', header_2020, widths_2020, columns_2020, column_names_2020, 2020, True)
df2020_yes = process_file(f'{data_folder}/mass2020.txt', header_2020, widths_2020, columns_2020, column_names_2020, 2020, False)

scenarios = [(df2016_yes, df2016_no, f'{data_folder}/df2016_2016_noyes.csv'), 
             (df2020_no, df2016_no, f'{data_folder}/df2016_2020_nono.csv'),     
             (df2020_yes, df2016_yes, f'{data_folder}/df2016_2020_yesyes.csv')]

for df1, df2, output_file in scenarios:
    calculate_difference(df1, df2, output_file)


file = "data/df2016_2016_noyes.csv"
d2016_2016_noyes = pd.read_csv(file, delimiter=';')
plot_data(d2016_2016_noyes, 'Diff_nuclear_mass', '(MeV)', 'nuclear_mass_expteo_dif_#.pdf',
          'Data processing plots', cmap='seismic', vmin=d2016_2016_noyes['Diff_nuclear_mass'].min(), vmax=d2016_2016_noyes['Diff_nuclear_mass'].max())
