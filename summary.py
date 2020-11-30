#!/usr/bin/env python


import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# In[2]:


columns = ['day', 'hospitalizeNum', 'isolateNum', 'quarantineNum', 'confineNum',
       'free', 'CurrentHealthy', 'CurrentInfected', 'CurrentEffective','CurrentSusceptible', 'CurrentIncubation', 'CurrentDiscovered', 'CurrentCritical', 'CurrentRecovered',
       'AccDiscovered', 'AccCritical', 'AccAcquaintance', 'AccStranger', 'measurement'] 

# Here are some consts for metrics
I_threshold = 500
Q_threshold = 10000
Q_weight = 1
time = 59 # The last day of our simulation


# In[3]:


def process_file(file):
    # Read data

    df = pd.read_csv(file, sep=',', engine='python', header=None)
    df.columns = columns
    hos=df['hospitalizeNum'].values
    iso=df['isolateNum'].values
    quar=df['quarantineNum'].values
    conf=df['confineNum'].values
    health=df['CurrentHealthy'].values
    x=np.arange(60)
    plt.plot(x,health)
    plt.plot(x, iso)
    plt.plot(x, quar)
    plt.plot(x, conf)
    plt.plot(x, hos)
    plt.legend(['health-num','iso-num','quar-num','conf-num','hos-num'],loc='upper right')
    #plt.show()
    # Accumulate 
    for item in ['hospitalizeNum', 'isolateNum', 'quarantineNum', 'confineNum']:
        sum = 0
        list_sum = []
        ind = 0
        l = len(df[item])
        while ind < l:
            sum += df[item][ind]
            list_sum.append(sum)
            if df["day"][ind] == time:
                sum = 0
            ind += 1
        df["sum_"+item] = np.array(list_sum)
    return df


# In[4]:


def get_I_Q(df, time):
    # Get I and Q value at the given time

    df_sub = df[df["day"]==time]
    I = df_sub["CurrentInfected"].mean()

    inHospital_mean = df_sub["sum_hospitalizeNum"].mean()
    isolateNum_mean = df_sub["sum_isolateNum"].mean()
    confineNum_mean = df_sub["sum_confineNum"].mean()
    quarantineNum_mean = df_sub["sum_quarantineNum"].mean()
    Q = 1 *  inHospital_mean + 0.5 * isolateNum_mean+ 0.3 * quarantineNum_mean + 0.2 * confineNum_mean

    return I, Q


# In[5]:


def get_least_Q_score(I, Q, I_threshold):
    score = np.copy(Q)
    score[I > I_threshold] = 1e6

    return score


# In[6]:


def get_exp_score(I, Q, I_threshold, Q_threshold, Q_weight):
    I_score = np.exp(I/I_threshold)
    Q_score = Q_weight * (np.exp(Q/ Q_threshold))
    
    return I_score + Q_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eva_txt", type=string, help='evaluation text',default=“cnt_scenario1.txt")
    args = parser.parse_args()
    path=args.eva_txt                   
    df = process_file(path)
    I, Q = get_I_Q(df, time)
    print(I,Q)
    least_Q_score = get_least_Q_score(I, Q, I_threshold)
    exp_score = get_exp_score(I, Q, I_threshold, Q_threshold, Q_weight)
    print(least_Q_score, exp_score)
