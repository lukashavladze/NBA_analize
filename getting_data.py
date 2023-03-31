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
rows = list(range(1, 688))
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


# pd.set_option('display.max_rows', None)




boston_top = df[(df['Tm'] == 'BOS') & (df['PTS'] > 11)]

boston = df[(df['Tm'] == 'BOS')]
atlanda = df[(df['Tm'] == 'ATL')]
houston = df[(df['Tm'] == 'HOU')]
memphis = df[(df['Tm'] == 'MEM')]
denver = df[(df['Tm'] == 'DEN')]
sacramento = df[(df['Tm'] == 'SAC')]
phoenix = df[(df['Tm'] == 'PHO')]
los_angeles_clippers = df[(df['Tm'] == 'LAC')]
golden_state = df[(df['Tm'] == 'GSW')]
minesota = df[(df['Tm'] == 'MIN')]
new_orlean = df[(df['Tm'] == 'NOP')]
los_angeles_lakers = df[(df['Tm'] == 'LAL')]
oklahoma = df[(df['Tm'] == 'OKC')]
dallas = df[(df['Tm'] == 'DAL')]
utah = df[(df['Tm'] == 'UTA')]
portland = df[(df['Tm'] == 'POR')]
san_antonio = df[(df['Tm'] == 'SAS')]
milwaukee = df[(df['Tm'] == 'MIL')]
philadelphia = df[(df['Tm'] == 'PHI')]
cleveland = df[(df['Tm'] == 'CLE')]
new_york = df[(df['Tm'] == 'NYK')]
brooklyn = df[(df['Tm'] == 'BRK')]
miami = df[(df['Tm'] == 'MIA')]
toronto = df[(df['Tm'] == 'TOR')]
chicago = df[(df['Tm'] == 'CHI')]
washington = df[(df['Tm'] == 'WAS')]
indiana = df[(df['Tm'] == 'IND')]
orlando = df[(df['Tm'] == 'ORL')]
detroit = df[(df['Tm'] == 'DET')]
charlotte = df[(df['Tm'] == 'CHO')]

print(milwaukee)

# sns.scatterplot(x='PTS', y='Player',  hue='Pos', palette='bright', data=boston)
# plt.show()
sns.scatterplot(x='PTS', y='Player',  hue='Pos', palette='bright', data=milwaukee)
plt.show()

# sns.lmplot(data=boston, x='PTS', y='BLK', hue='Pos')
# plt.show()


# print(boston_top)





