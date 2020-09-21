import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
#%matplotlib notebook


def read_ground_truth(filename):
    df=pd.read_csv(filename)
    print(df.columns)
    #print('######')
    locations=set(df["LocationID"])
    #print(locations)

    df["DateTime"]=pd.to_datetime(df["DateTime"])
    #print(df["DateTime"])
    df["Value"].astype('float')

    loc2values={}

    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        #print('$$$$$$$$$$$')
        #print(sub_df.head())

        loc2values[loc]=sub_df

    return df, loc2values

def draw_df(loc2values):
    #plt.plot(df["DateTime"],df["Value"])
    plt.figure(0)
    locs=[]
    for loc in loc2values:
        plt.plot(loc2values[loc]["DateTime"], loc2values[loc]["Value"])
        locs.append(loc)
    plt.legend(locs, loc='upper left')
    plt.show()

if __name__=='__main__':
    #df, loc2values=read_ground_truth(r'C:\TC\StreamFlow\data\2016wholeyear')
    df, loc2values=read_ground_truth(r'/vc_data/zhuwe/jupyter_sever_logs/tc_sfr/data/2016wholeyear')
    draw_df(loc2values)

