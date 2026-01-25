import matplotlib.pyplot as plt

def fontsizes(config):
    plt.rcParams.update({
        'font.size': config['fontsizes']['font_size'],
        'axes.titlesize': config['fontsizes']['axes_title_size'],
        'axes.labelsize': config['fontsizes']['axes_label_size'],
        'xtick.labelsize': config['fontsizes']['xtick_labelsize'],
        'ytick.labelsize': config['fontsizes']['ytick_labelsize'],
        'legend.fontsize': config['fontsizes']['legend_fontsize'],
        'figure.titlesize': config['fontsizes']['figure_title_size'],
        'font.family': 'serif',
        'font.serif': ['STIX', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
    })
    return
