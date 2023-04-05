import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


url2 = 'https://www.basketball-reference.com/leagues/NBA_2023_per_game.html#per_game_stats'
data = pd.read_html(url2)
headers = (data[0].head(0))
headers_list = []
for i in headers.columns:
    headers_list.append(i)
# print(headers_list)
npdata = np.array(data[0])
rows = list(range(1, 698))
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


# pd.set_option('display.max_columns', None)

boston_top = df[(df['Tm'] == 'BOS') & (df['PTS'] > 11)]

boston = df[(df['Tm'] == 'BOS')]
atlanta = df[(df['Tm'] == 'ATL')]
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


# print(milwaukee[milwaukee['Pos'] == 'SG'])

def show_top_players(team_name, position, sorted_by, numb_players=2):
    team_pos = (team_name[team_name['Pos'] == position].sort_values(sorted_by, ascending=False).head(numb_players))
    print(team_pos)


# test = boston[(boston['Pos'] == 'PG') | (boston['Pos'] == 'SG')]
# print(test)
def every_position_player_bypts(team):
    pg = show_top_players(team, 'PG', 'PTS', 2)
    sg = show_top_players(team, 'SG', 'PTS', 2)
    c = show_top_players(team, 'C', 'PTS', 2)
    pf = show_top_players(team, 'PF', 'PTS', 2)
    sf = show_top_players(team, 'SF', 'PTS', 2)
    print(pg, sg, c, pf, sf)

every_position_player_bypts(milwaukee)


# bos_PG = (boston[boston['Pos'] == 'PG'].sort_values('PTS', ascending=False).head(2))
# print(bos_PG)

# sns.scatterplot(x='PTS', y='Player',  hue='Pos', palette='bright', data=milwaukee.sort_values('Pos'))
# plt.show()

# a = Line2D([], [], color='blue', label='PG')
# b = Line2D([], [], color='green', label='SG')
# c = Line2D([], [], color='purple', label='PF')
# d = Line2D([], [], color='red', label='SF')
# e = Line2D([], [], color='yellow', label='C')
# plt.legend(handles=[a, b, c, d, e])
# sns.histplot(x='PTS', y='Player', hue='Pos', palette=(a._color, b._color, c._color, d._color, e._color), data=milwaukee.sort_values('PTS'))
# plt.show()

# sns.lmplot(data=boston, x='PTS', y='BLK', hue='Pos')
# plt.show()
