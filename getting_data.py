import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


url2 = 'https://www.basketball-reference.com/leagues/NBA_2023_per_game.html#per_game_stats'
data = pd.read_html(url2)
headers = (data[0].head(0))
headers_list = []
for i in headers.columns:
    headers_list.append(i)
# print(headers_list)
npdata = np.array(data[0])
rows = list(range(1, 687))
columns = list(range(1, 31))
df = pd.DataFrame(npdata, rows, headers_list)
df = df.drop_duplicates()
df.drop(labels=25, inplace=True)
# converting column data str into float
df['Rk'] = df['Rk'].astype(int)
df['Age'] = df['Age'].astype(int)
df['G'] = df['G'].astype(float)
df['GS'] = df['GS'].astype(float)
df['MP'] = df['MP'].astype(float)
df['FG'] = df['FG'].astype(float)
df['FGA'] = df['FGA'].astype(float)
df['FG%'] = df['FG%'].astype(float)
df['3P'] = df['3P'].astype(float)
df['3PA'] = df['3PA'].astype(float)
df['3P%'] = df['3P%'].astype(float)
df['2P'] = df['2P'].astype(float)
df['2PA'] = df['2PA'].astype(float)
df['2P%'] = df['2P%'].astype(float)
df['eFG%'] = df['eFG%'].astype(float)
df['FT'] = df['FT'].astype(float)
df['FTA'] = df['FTA'].astype(float)
df['FT%'] = df['FT%'].astype(float)
df['ORB'] = df['ORB'].astype(float)
df['DRB'] = df['DRB'].astype(float)
df['TRB'] = df['TRB'].astype(float)
df['AST'] = df['AST'].astype(float)
df['STL'] = df['STL'].astype(float)
df['BLK'] = df['BLK'].astype(float)
df['TOV'] = df['TOV'].astype(float)
df['PF'] = df['PF'].astype(float)
df['PTS'] = df['PTS'].astype(float)


pd.set_option('display.max_rows', None)

sns.barplot(x='Tm', y='PTS', data=df)
plt.show()








