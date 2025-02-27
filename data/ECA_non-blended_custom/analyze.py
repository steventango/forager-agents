# %%
import pandas as pd

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
import seaborn as sns
import matplotlib.pyplot as plt

# plot 10 year windows:
n_windows = (df["date"].dt.year.max() - df["date"].dt.year.min()) // 10
print(n_windows)
fig, axs = plt.subplots(n_windows, 1, figsize=(10, 10 * n_windows))
for ax, start_year in zip(axs, range(df["date"].dt.year.min(), df["date"].dt.year.max(), 10)):
    end_year = start_year + 10
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-01-01')
    sub_df = df[(df["date"] >= start_date) & (df["date"] < end_date)]
    sub_df = sub_df.reset_index()
    print(len(sub_df))
    sns.lineplot(data=sub_df, x='index', y='normalized_mean_temperature', ax=ax)
    ax.set_title(f'Normalized Mean Temperature from {start_year} to {end_year}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Mean Temperature')
plt.tight_layout()
plt.show()

# sns.lineplot(data=df, x='date', y='normalized_mean_temperature')

# %%
