#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime, timedelta
import tqdm
import csv


#%%
def hit_pitch_PA(data, n, timeframes, pa_stats):
    outcomes_dict = dict(zip(pa_stats, ([] for _ in pa_stats)))
    h_tf_stats_dict = dict(zip(pa_stats, ([] for _ in pa_stats)))
    p_tf_stats_dict = dict(zip(pa_stats, ([] for _ in pa_stats)))
   
    data_start_dt = datetime.strptime(data['game_date'].iloc[-1], '%Y-%m-%d').date()
    PA_df = data.dropna(subset = ['events'])
    n_PA = len(PA_df)
    # max_idx is last idx in PA_df that provides enough previous data to fullfill largest timeframe
    
    max_idx = n_PA-1
    for i in range(n_PA):
        max_idx -= 1
        if datetime.strptime(PA_df['game_date'].iloc[max_idx], '%Y-%m-%d').date() > data_start_dt + timedelta(days=timeframes[-1]):
            break
    

    # generate list of n random indices to 
    idxs = random.sample(range(max_idx), n)
  
    for j in tqdm.tqdm(range(n)):
        i = idxs[j]
        date = datetime.strptime(PA_df['game_date'].iloc[i], '%Y-%m-%d').date()


        hitter_id = int(PA_df['batter'].iloc[i])
        hitter_data = PA_df[PA_df['batter'].astype(int) == hitter_id]
        hitter_timeframe_rows = []
        for index, row in hitter_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                hitter_timeframe_rows.append(row)
        hitter_timeframe_data = pd.DataFrame(hitter_timeframe_rows)
       

        pitcher_id = int(PA_df['pitcher'].iloc[i])
        pitcher_data = PA_df[PA_df['pitcher'].astype(int) == pitcher_id]
        pitcher_timeframe_rows = []
        for index, row in pitcher_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                pitcher_timeframe_rows.append(row)
        pitcher_timeframe_data = pd.DataFrame(pitcher_timeframe_rows)

        # add outcome of PA_df[i] to pitcher and hitter outcome dicionaries
        outcomes_dict = find_PA_outcome(outcomes_dict, PA_df, i, True)
        for stat in pa_stats:
            h_tf_stats_dict[stat].append(find_PA_time_stats(hitter_timeframe_data, stat, timeframes, date))
            p_tf_stats_dict[stat].append(find_PA_time_stats(pitcher_timeframe_data, stat, timeframes, date))

    return outcomes_dict, h_tf_stats_dict, p_tf_stats_dict
def find_PA_outcome(stats_dict, PA_df, idx, pa):
    if pa == True:
        for stat in stats_dict:
            # Find outcome of PA for stat
            if stat in ['walk', 'strikeout', 'hit_by_pitch']:
                if PA_df['events'].iloc[idx] == stat:
                    stats_dict[stat].append(1)
                else:
                    stats_dict[stat].append(0)
            elif stat in ['ground_ball', 'fly_ball', 'line_drive', 'popup'] :
                if not pd.isnull(PA_df['bb_type'].iloc[idx]):
                    if PA_df['bb_type'].iloc[idx] == stat:
                        stats_dict[stat].append(1)
                    else:
                        stats_dict[stat].append(0)
                else:
                    stats_dict[stat].append(0)
            else:
                if not pd.isnull(PA_df['estimated_woba_using_speedangle'].iloc[idx]):
                    stats_dict[stat].append(PA_df['estimated_woba_using_speedangle'].iloc[idx])
                else:
                    stats_dict[stat].append(PA_df['woba_value'].iloc[idx])
        
    else:
        stats_dict['launch_speed'].append(PA_df['launch_speed'].iloc[idx])
        stats_dict['launch_angle'].append(PA_df['launch_angle'].iloc[idx])
    return stats_dict
def find_PA_time_stats(player_timeframe_data, stat, time_frames, date):
    time_frame_stat_ctr = dict(zip(time_frames, (0 for _ in time_frames)))
    time_frame_PA_ctr = dict(zip(time_frames, (0 for _ in time_frames)))
    for timeframe in time_frames:
        start_dt =  date - timedelta(days=timeframe)
        for index, row in player_timeframe_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() >= start_dt:
                time_frame_PA_ctr[timeframe] += 1
                if stat in ['walk', 'strikeout', 'hit_by_pitch']:
                    if row['events'] == stat:
                        time_frame_stat_ctr[timeframe] += 1
                elif stat in ['ground_ball', 'fly_ball', 'line_drive', 'popup'] :
                    if not pd.isnull(row['bb_type']):
                        if row['bb_type'] == stat:
                            time_frame_stat_ctr[timeframe] += 1
                else:
                    if not pd.isnull(row['estimated_woba_using_speedangle']):
                        time_frame_stat_ctr[timeframe] += row['estimated_woba_using_speedangle']
                    else:
                        time_frame_stat_ctr[timeframe] += row['woba_value']
    time_frame_stats = []
    for time in time_frames:
        if time_frame_PA_ctr[time] == 0:
            time_frame_stats.append([0,0])
        else:
            time_frame_stats.append([time_frame_stat_ctr[time]/time_frame_PA_ctr[time], time_frame_PA_ctr[time]])
    return time_frame_stats
def find_bb_time_stats(player_timeframe_data, stat, time_frames, date):
    time_frame_stat_ctr = dict(zip(time_frames, (0 for _ in time_frames)))
    time_frame_PA_ctr = dict(zip(time_frames, (0 for _ in time_frames)))
    for timeframe in time_frames:
        start_dt =  date - timedelta(days=timeframe)
        for index, row in player_timeframe_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() >= start_dt:
                time_frame_PA_ctr[timeframe] += 1
                time_frame_stat_ctr[timeframe] += row[stat]
    time_frame_stats = []
    for time in time_frames:
        if time_frame_PA_ctr[time] == 0:
            time_frame_stats.append([0, 0])
        else:
            time_frame_stats.append([time_frame_stat_ctr[time]/time_frame_PA_ctr[time], time_frame_PA_ctr[time]])
    return time_frame_stats
def hit_pitch_bb(data, n, timeframes, bb_stats):
    outcomes_dict = dict(zip(bb_stats, ([] for _ in bb_stats)))
    h_tf_stats_dict = dict(zip(bb_stats, ([] for _ in bb_stats)))
    p_tf_stats_dict = dict(zip(bb_stats, ([] for _ in bb_stats)))
   
    
    bb_df = data.dropna(subset = ['type', 'launch_speed', 'launch_angle'])
    bb_df = bb_df[bb_df['type'] == 'X']
    data_start_dt = datetime.strptime(bb_df['game_date'].iloc[-1], '%Y-%m-%d').date()
    n_bb = len(bb_df)
    max_idx = n_bb-1
    for i in range(n_bb):
        max_idx -= 1
        if datetime.strptime(bb_df['game_date'].iloc[max_idx], '%Y-%m-%d').date() > data_start_dt + timedelta(days=timeframes[-1]):
            break

    # generate list of n random indices to 
    idxs = random.sample(range(max_idx), n)
    for i in tqdm.tqdm(idxs):
        date = datetime.strptime(bb_df['game_date'].iloc[i], '%Y-%m-%d').date()


        hitter_id = int(bb_df['batter'].iloc[i])
        hitter_data = bb_df[bb_df['batter'].astype(int) == hitter_id]
        hitter_timeframe_rows = []
        for index, row in hitter_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                hitter_timeframe_rows.append(row)
        hitter_timeframe_data = pd.DataFrame(hitter_timeframe_rows)
       

        pitcher_id = int(bb_df['pitcher'].iloc[i])
        pitcher_data = bb_df[bb_df['pitcher'].astype(int) == pitcher_id]
        pitcher_timeframe_rows = []
        for index, row in pitcher_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                pitcher_timeframe_rows.append(row)
        pitcher_timeframe_data = pd.DataFrame(pitcher_timeframe_rows)

        # add outcome of PA_df[i] to pitcher and hitter outcome dicionaries
        outcomes_dict = find_PA_outcome(outcomes_dict, bb_df, i, False)
        for stat in bb_stats:
            h_tf_stats_dict[stat].append(find_bb_time_stats(hitter_timeframe_data, stat, timeframes, date))
            p_tf_stats_dict[stat].append(find_bb_time_stats(pitcher_timeframe_data, stat, timeframes, date))

    return outcomes_dict, h_tf_stats_dict, p_tf_stats_dict
#%%
data = pd.read_csv('pitch_by_pitch_data.csv')
#%%
timeframes = [7, 15, 45, 90]
pa_stats = ['walk', 'strikeout', 'hit_by_pitch', 'ground_ball', 'fly_ball', 'line_drive', 'popup', 'xwoba']
bb_stats = ['launch_speed', 'launch_angle']
n = 10000

#%%
print('Generating PA data...')
pa_outcomes_dict, pa_h_tf_stats_dict, pa_p_tf_stats_dict = hit_pitch_PA(data, n, timeframes, pa_stats)
print('Generating BB data...')
bb_outcomes_dict, bb_h_tf_stats_dict, bb_p_tf_stats_dict = hit_pitch_bb(data, n, timeframes, bb_stats)


# %%
pa_h_tf_stats_dict['walk']

#%%

columns = ['p7', 'p7#', 'h7', 'h7#', 'p15', 'p15#', 'h15', 'h15#', 'p45', 'p45#', 'h45', 'h45#', 'p90', 'p90#', 'h90', 'h90#', 'Result']

for stat in pa_outcomes_dict:
    data_dict = dict(zip(columns, ([] for _ in columns)))
    for i in range(n):
        for j in range(len(timeframes)):
            data_dict[columns[4 * j]].append(pa_p_tf_stats_dict[stat][i][j][0])
            data_dict[columns[4 * j + 1]].append(pa_p_tf_stats_dict[stat][i][j][1])
            data_dict[columns[4 * j + 2]].append(pa_h_tf_stats_dict[stat][i][j][0])
            data_dict[columns[4 * j + 3]].append(pa_h_tf_stats_dict[stat][i][j][1])
        data_dict['Result'].append(pa_outcomes_dict[stat][i])
  

    df = pd.DataFrame.from_dict(data_dict)
    
    df.to_csv(stat + '.csv')
        
    


# %%

for stat in bb_outcomes_dict:
    data_dict = dict(zip(columns, ([] for _ in columns)))
    for i in range(n):
        for j in range(len(timeframes)):
            data_dict[columns[4 * j]].append(bb_p_tf_stats_dict[stat][i][j][0])
            data_dict[columns[4 * j + 1]].append(bb_p_tf_stats_dict[stat][i][j][1])
            data_dict[columns[4 * j + 2]].append(bb_h_tf_stats_dict[stat][i][j][0])
            data_dict[columns[4 * j + 3]].append(bb_h_tf_stats_dict[stat][i][j][1])
        data_dict['Result'].append(bb_outcomes_dict[stat][i])
  

    df = pd.DataFrame.from_dict(data_dict)
    
    df.to_csv(stat + '.csv')

# %%
