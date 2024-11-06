import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Extracting data from WS4 file
with open('data/WS4.txt', 'r') as file:
    for i in range(29):
        if i == 27:  
            header = file.readline().strip().split() 
        else:
            next(file) 
    data = [line.strip() for line in file]

data_rows = [line.split() for line in data]
dfWS4= pd.DataFrame(data_rows, columns=header)  
dfWS4.to_csv('data/WS4_cleaned.csv', index=False, header=True, sep=';')
print("WS4 dataset: \n", dfWS4.head(), "\n")


# Function to extract data from AME2020 and AME2016 files
def process_file(filename, header, widths, columns, column_names, year):
    df = pd.read_fwf(
        filename,
        usecols=columns,
        names=column_names,
        widths=widths,
        header=header,
        index_col=False
    )
    df = df.replace({'#': ''}, regex=True) 
    df = df[(df['N'] >= 8) & (df['N'] < 180) & (df['Z'] >= 8) & (df['Z'] < 120)] # Restrictions for our study case 

    df['Z'] = df['Z'].astype(int) 
    df['A'] = df['A'].astype(int)
    df['delta'] = np.where((df['Z'] % 2 == 0) & (df['N'] % 2 == 0), -1,  # Z and N even
                   np.where(df['A'] % 2 == 1, 0,  # A odd
                   np.where((df['Z'] % 2 == 1) & (df['N'] % 2 == 1), 1, np.nan)))  # Z and N odds
    df['delta'] = df['delta'].astype(int)
    df['delta_I4'] = ((-1)**df['N']+(-1)**df['Z'])/2
    df['delta_I4'] = df['delta_I4'].astype(int)

    # Theoretical model
    av = 15.8  # MeV
    aS = 18.3
    ac = 0.66
    aA = 23.2
    ap = 11.2

    A = df['A']
    Z = df['Z']
    N = df['N']
    delta = df['delta']

    df['bind_ene'] = df['bind_ene'].astype(float)/1000 # Everything is in MeV
    df['bind_ene_total'] = df['bind_ene']*df['A'] 
    df['bind_ene_teo'] = (av*A-aS*A**(2/3)-ac*Z**2*A**(-1/3)-aA*(A-2*Z)**2/A-ap*delta*A**(-1/2))/A  
    df['bind_ene_teo_total'] = df['bind_ene_teo']*df['A']
    df['Diff_bind_ene'] = df['bind_ene'] - df['bind_ene_teo']
    df['Diff_bind_ene_total'] = df['bind_ene_total'] - df['bind_ene_teo_total']
 
    uma = 931.4936 
    m_e = 0.510998928 
    m_p = 938.27208816
    m_n = 939.565378

    df['A2'] = df['A2'].astype(str)
    df['atomic_mass'] = df['atomic_mass'].astype(str)
    df['atomic_mass'] = df['A2'] + df['atomic_mass']
    df['atomic_mass'] = pd.to_numeric(df['atomic_mass'], errors='coerce')
    df['atomic_mass'] = df['atomic_mass'].astype(float)
    df['atomic_mass'] = df['atomic_mass']/(10**6) #u

    df['atomic_mass_unc'] = pd.to_numeric(df['atomic_mass_unc'], errors='coerce')
    df['atomic_mass_unc'] = df['atomic_mass_unc'].astype(float)
    df['atomic_mass_unc'] = df['atomic_mass_unc']/(10**6) #u

    df['atomic_mass_teo'] = Z*m_p + N*m_n - df['bind_ene_teo_total']
    
    df['atomic_mass_2'] = Z*m_p + N*m_n - df['bind_ene_total']   

    df['B_e'] = (14.4381*(Z**2.39) + 1.55468*(10**-6)*(Z**5.35))*(10**-6)

    df['M_N_teo'] = df['atomic_mass_teo'] - Z*m_e + df['B_e']

    df['M_N_exp'] = df['atomic_mass_2'] - Z*m_e + df['B_e']
    
    df['Diff_masses'] = df['M_N_exp'] - df['M_N_teo']
    df.to_csv(f'data/mass{year}_cleaned.csv', sep=';', index=False)
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

df2020 = process_file('data/mass2020.txt', header_2020, widths_2020, columns_2020, column_names_2020, 2020)
print("AME2020 dataset: \n", df2020.head(), "\n")

df2016 = process_file('data/mass2016.txt', header_2016, widths_2016, columns_2016, column_names_2016, 2016)
print("AME2016 dataset: \n", df2016.head(), "\n")

rmse_2016_bind_ene = np.sqrt(np.mean(df2016['Diff_bind_ene'] ** 2))
print('RMSE liquid droplet model (2016) binding energy per nucleon: ', rmse_2016_bind_ene, 'MeV')


#Plots of AME2016 dataset:
binding_plots_folder = 'Binding energy plots'
if not os.path.exists(binding_plots_folder):
    os.makedirs(binding_plots_folder)

#Plot of the theoretical binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2016['N'], df2016['Z'], c=df2016['bind_ene_teo'], cmap='jet', edgecolor='None',
                      s=12, vmin=df2016['bind_ene_teo'].min(), vmax=df2016['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Theoretical binding energy per nucleon AME2016')
plt.savefig(os.path.join(binding_plots_folder, 'bind_teo_per_nucleon.png'))
plt.show()

#Plot of the experimental binding energy as a function of Z and N
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df2016['N'], df2016['Z'], c=df2016['bind_ene'], cmap='jet', edgecolor='None',
                      s=12, vmin=df2016['bind_ene_teo'].min(), vmax=df2016['bind_ene_teo'].max())
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Experimental binding energy per nucleon AME2016')
plt.savefig(os.path.join(binding_plots_folder, 'bind_exp_per_nucleon.png'))
plt.show()

#Plot of the difference between theoretical and experimental binding energies
plt.figure(figsize=(10, 6))
norm = TwoSlopeNorm(vmin=df2016['Diff_bind_ene'].min(), vcenter=0, vmax=df2016['Diff_bind_ene'].max())
scatter = plt.scatter(df2016['N'], df2016['Z'], c=df2016['Diff_bind_ene'],
                      cmap='seismic', norm=norm, edgecolor='None', s=12)
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Difference exp-teo binding energy per nucleon AME2016')
plt.savefig(os.path.join(binding_plots_folder, 'bind_teoexp_dif_per_nucleon.png'))
plt.show()

#3D plot of the difference between theoretical and experimental binding energies
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')  
norm = TwoSlopeNorm(vmin=df2016['Diff_bind_ene'].min(), vcenter=0, vmax=df2016['Diff_bind_ene'].max())
scatter = ax.scatter(df2016['Z'], df2016['N'], df2016['Diff_bind_ene'], c=df2016['Diff_bind_ene'],
                     cmap='seismic', norm=norm, edgecolor='None', s=12)
ax.set_xlabel('Z')
ax.set_ylabel('N')
plt.title('3D Difference exp-teo binding energy per nucleon AME2016 ')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('(MeV)')
plt.savefig(os.path.join(binding_plots_folder, 'bind_teoexp_dif_3D_per_nucleon.png'))
plt.show()


#Nuclear shell gaps (\Delta_{2n} and \Delta_{2p})
def calculate_shell_gaps(df, element, axis, type, column, year):
    #Neutrons--> element = n, axis = Z; Protons--> element = p, axis = N
    #type = exp or teo; column = 'bind_ene_total' or 'bind_ene_teo_total'
    df[f'bind_ene_{element}+2_{type}'] = df.groupby(axis)[column].shift(-2)
    df[f'bind_ene_{element}-2_{type}'] = df.groupby(axis)[column].shift(2)
    df[f'delta_2{element}_{type}'] = df[f'bind_ene_{element}-2_{type}'] - 2 * df[column] + df[f'bind_ene_{element}+2_{type}']
    df.to_csv(f'data/mass{year}_cleaned.csv', sep=';', index=False)
    return df

def plot_shell_gaps(df, gap_col, type, title, filename, binding_plots_folder, vmin, vmax, xlim, ylim): #gap_col=delta_2n or delta_2p
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
    plt.title("{} {}".format(title, type))
    plt.savefig(os.path.join(binding_plots_folder, filename))
    plt.show()

df2016 = calculate_shell_gaps(df2016, 'n', 'Z', 'exp', 'bind_ene_total', 2016) 
df2016 = calculate_shell_gaps(df2016, 'p', 'N', 'exp', 'bind_ene_total', 2016) 
df2016 = calculate_shell_gaps(df2016, 'n', 'Z', 'teo', 'bind_ene_teo_total', 2016) 
df2016 = calculate_shell_gaps(df2016, 'p', 'N', 'teo', 'bind_ene_teo_total', 2016) 

min_value = min(df2016['delta_2n_exp'].min(), df2016['delta_2p_exp'].min(),
                df2016['delta_2n_teo'].min(), df2016['delta_2p_teo'].min()) #Same colorbar range for both plots
max_value = max(df2016['delta_2n_exp'].max(), df2016['delta_2p_exp'].max(),
                df2016['delta_2n_teo'].max(), df2016['delta_2p_teo'].max())

xlim = (min(df2016['N'].min(), 0), max(df2016['N'].max() + 10, 0)) #Same limits for both plots
ylim = (0, df2016['Z'].max() + 10)  

plot_shell_gaps(df2016, 'delta_2n', 'exp', 'Neutron shell gaps', 'neutron_shell_gaps_exp.png', binding_plots_folder,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2p', 'exp', 'Proton shell gaps', 'proton_shell_gaps_exp.png', binding_plots_folder,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2n', 'teo', 'Neutron shell gaps', 'neutron_shell_gaps_teo.png', binding_plots_folder,
                min_value, max_value, xlim=xlim, ylim=ylim)
plot_shell_gaps(df2016, 'delta_2p', 'teo', 'Proton shell gaps', 'proton_shell_gaps_teo.png', binding_plots_folder,
                min_value, max_value, xlim=xlim, ylim=ylim)


# Nuclear masses
rmse_2016_nuclear_mass = np.sqrt(np.mean(df2016['Diff_masses'] ** 2))
print('RMSE liquid droplet model (2016) nuclear masses: ', rmse_2016_nuclear_mass, 'MeV')

#Plot of the difference between theoretical and experimental nuclear masses
plt.figure(figsize=(10, 6))
norm = TwoSlopeNorm(vmin=df2016['Diff_masses'].min(), vcenter=0, vmax=df2016['Diff_masses'].max())
scatter = plt.scatter(df2016['N'], df2016['Z'], c=df2016['Diff_masses'],
                      cmap='seismic', norm=norm, edgecolor='None', s=12)
cbar = plt.colorbar(scatter)
cbar.set_label('(MeV)')
magic_numbers = [8, 20, 28, 50, 82, 126]
for magic in magic_numbers:
    plt.axvline(x=magic, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=magic, color='gray', linestyle='--', linewidth=0.5)
plt.xticks(magic_numbers)
plt.yticks(magic_numbers)
plt.xlabel('N')
plt.ylabel('Z') 
plt.title('Difference exp-teo in masses AME2016')
plt.savefig(os.path.join(binding_plots_folder, 'masses_teoexp_dif.png'))
plt.show()