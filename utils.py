import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def get_data(subject):
    anno_file = pd.read_csv(r'/Users/shiyuanduan/Documents/PainStudy/FINAL times lab baseline for ML 9_5_23.csv', encoding='utf-8')
    anno_file.set_index('Subject N', inplace=True)
    # anno_file.loc['S1']['baseline HR']
    file_path = f'./Pain_Data_w_Freq/{subject}_Summary.csv'

    # cols = 'C_Active(g),L_Active(g),itp_F0(Hz),itp_Po(dB),itp_Temp(c),itp_HR(bpm),itp_RR(bpm),itp_HRV'.split(',')

    # cols = ['L_Active', 'itp_Po', 'itp_Temp', 'itp_HR', 'itp_RR']
    data = pd.read_csv(file_path)
    data = data.set_index('time(s)')


    anno = anno_file.loc[f'S{subject}']

    base_hr = anno['baseline HR']
    base_rr = anno['baseline RR']
    start = anno['Sensor Lab time (s)']
    end = anno['Sensor lab stop 9s)']
    # data['HR_interval'] = data['itp_HR'] * 1000 / 60
    data['itp_HR(bpm)'] -= base_hr
    data['itp_RR(bpm)'] -= base_rr
    start = anno['Sensor Lab time (s)']
    end = anno['Sensor lab stop 9s)']
    
    data = data.ffill().bfill()
    return {'data_np':data.values, 'data_df':data, 'lab_draw_start':start, 'lab_draw_end':end, 'base_hr':base_hr, 'base_rr':base_rr}



def prolonged(slice):
    return slice.std() <= 30

def decision_tree(slice):
    if slice.max()>450:
        return 2

    if slice.mean() < 50:
        return 0
    
    if prolonged(slice):
        return 1
    

    else:
        return 0
    

def label_pain(df):
    df['pain_score'] = df['itp_F0(Hz)'].rolling(30).apply(decision_tree)
    return df

def acc_decision_tree(slice):
    if slice.max() > 0.2:
        return 2
    elif slice.max()>0.1 and slice.max()<0.2:
        return 1
    else:
        return 0
    
def label_acc(df):
    df['acc_score'] = df['L_Active(g)'].rolling(30).apply(acc_decision_tree)
    return df.fillna(0)

def hr_decision_tree(hr, hr_baseline):
    def in_range(x, low, high):
        return x>low and x < high

    if hr.max() > 0.2*hr_baseline:
        return 2
    elif in_range(hr.max(), 0.1*hr_baseline, 0.2*hr_baseline):
        return 1
    else:
        return 0
    
def label_vitals_hr(df, hr_baseline):

    df['hr_score'] = df['itp_HR(bpm)'].rolling(window=30).apply(lambda x: hr_decision_tree(x, hr_baseline), raw=False)

    return df.fillna(0)


def rr_decision_tree(rr, rr_baseline):
    def in_range(x, low, high):
        return x>low and x < high

    if rr.max() > 0.2*rr_baseline:
        return 2
    elif in_range(rr.max(), 0.1*rr_baseline, 0.2*rr_baseline):
        return 1
    else:
        return 0
    
def label_vitals_rr(df, rr_baseline):

    df['rr_score'] = df['itp_RR(bpm)'].rolling(window=30).apply(lambda x: hr_decision_tree(x, rr_baseline), raw=False)

    return df.fillna(0)

def label_data(sub):
    data = get_data(sub)
    df, base_hr, base_rr = data['data_df'], data['base_hr'], data['base_rr']

    # label all vitals
    df = label_pain(df)
    df = label_acc(df)
    df = label_vitals_hr(df, base_hr)
    df = label_vitals_rr(df, base_rr)
    df['vital_score'] = df[['hr_score','rr_score']].max(axis=1)
    df['score_sum'] = df[['vital_score', 'pain_score', 'acc_score']].sum(axis=1)
    return df


def plot_distribution(subject):
    anno_file = pd.read_csv(r'./Users/shiyuanduan/Documents/PainStudy/FINAL times lab baseline for ML 9_5_23.csv', encoding='utf-8')
    anno_file.set_index('Subject N', inplace=True)
    
    file_path = f'/Users/shiyuanduan/Documents/PainStudy/Pain_Data_w_Freq/{subject}_Summary.csv'
    cols = ['L_Active', 'itp_Po', 'itp_Temp', 'itp_HR', 'itp_RR']
    data = pd.read_csv(file_path)
    data = data.set_index('time(s)')
    data = data[cols]
    data['label'] = np.zeros(len(data))

    anno = anno_file.loc[f'S{subject}']

    base_hr = anno['baseline HR']
    base_rr = anno['baseline RR']
    start = anno['Sensor Lab time (s)']
    end = anno['Sensor lab stop 9s)']  # Please check this key, it seems to be a typo

    data['itp_HR'] -= data['itp_HR'].mean()
    data['itp_RR'] -= data['itp_RR'].mean()
    data.loc[start:end, 'label'] = 1
    
    # Normalize each feature
    for col in cols:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    # Creating subplots for violin plots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()  
    fig.delaxes(axes[-1])  

    # Drawing a violin plot of each feature in each label on subplots
    for i, col in enumerate(cols):
        sns.violinplot(x='label', y=col, data=data, ax=axes[i])
        axes[i].set_title(f'Violin plot of {col} by Label - sub: {subject}')

    plt.tight_layout()
    plt.show()

    # Creating line plots for each feature over time
    plt.figure(figsize=(15, 10))
    for col in cols:
        plt.plot(data.index, data[col], label=col)

    # Highlighting the area where label is 1 with a red background
    plt.axvspan(start, end, color='red', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.title(f'Line Plot of Normalized Features Over Time - sub: {subject}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the function with a specific subject number
# plot_distribution(1)  # Replace 1 with the actual subject number
