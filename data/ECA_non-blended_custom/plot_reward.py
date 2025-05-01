# %%
import pandas as pd
from PyExpPlotting.matplot import save, setDefaultConference, setFonts

# EUROPEAN CLIMATE ASSESSMENT & DATASET (ECA&D), file created on 27-02-2025
# THESE DATA CAN BE USED FREELY PROVIDED THAT THE FOLLOWING SOURCE IS ACKNOWLEDGED:

# Klein Tank, A.M.G. and Coauthors, 2002. Daily dataset of 20th-century surface
# air temperature and precipitation series for the European Climate Assessment.
# Int. J. of Climatol., 22, 1441-1453.
# Data and metadata available at http://www.ecad.eu

# FILE FORMAT (MISSING VALUE CODE IS -9999):

# 01-06 STAID: Station identifier
# 08-13 SOUID: Source identifier
# 15-22 DATE : Date YYYYMMDD
# 24-28 TG   : mean temperature in 0.1 &#176;C
# 30-34 Q_TG : Quality code for TG (0='valid'; 1='suspect'; 9='missing')

# This is the series (SOUID: 107054) of UNITED KINGDOM, HEATHROW (STAID: 1860)
# See file sources.txt for more info.



#  STAID, SOUID,    DATE,   TG, Q_TG
#   1860,107054,19600101,  106,    0
#   1860,107054,19600102,   61,    0


df = pd.read_csv('TG_SOUID100931.txt', skiprows=21)
df

#%%

df.columns = df.columns.str.strip()
#%%
df["mean_temperature"] = df["TG"] / 10
df = df[df["Q_TG"] == 0]
# normalize it between -1 and 1
df["normalized_mean_temperature"] = (df["mean_temperature"] - df["mean_temperature"].min()) / (df["mean_temperature"].max() - df["mean_temperature"].min()) * 2 - 1
df["date"] = pd.to_datetime(df["DATE"], format='%Y%m%d')
df["reward_hot"] = df["normalized_mean_temperature"]
df["reward_cold"] = -df["normalized_mean_temperature"]
# %%
import matplotlib.pyplot as plt
import numpy as np


def get_temperature(rewards: np.ndarray, clock: int, repeat: int) -> float:
    return rewards[clock // repeat % len(rewards)]

def reward_gen(rewards: np.ndarray, duration: int, repeat: int) -> np.ndarray:
    return np.array([get_temperature(rewards, clock, repeat) for clock in range(duration)])

def plot_reward(rewards: np.ndarray, duration: int, repeat: int, title: str):
    timeseries = reward_gen(rewards, duration, repeat)
    points = 500
    x = np.linspace(0, duration, points)
    timeseries = timeseries[::(len(timeseries) // points)]
    setFonts(20)
    plt.figure(figsize=(6, 6))
    plt.plot(x, timeseries, label='reward', color='black')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    plt.tight_layout()
    plt.savefig(title)
    plt.clf()

plot_reward(df["normalized_mean_temperature"].to_numpy(), int(1e6), 500, 'reward_timeseries_slow.pdf')
plot_reward(df["normalized_mean_temperature"].to_numpy(), int(1e6), 100, 'reward_timeseries_fast.pdf')
# %%
