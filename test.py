
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random
from datetime import datetime, timedelta

data = pd.read_csv('pitch_by_pitch_data.csv')


# # https://github.com/jldbc/pybaseball/tree/master/docs
# from pybaseball import statcast
# from pybaseball import batting_stats_bref
# from pybaseball import statcast_running_splits
# from pybaseball import statcast_batter
# from pybaseball import playerid_lookup
# from pybaseball import batting_stats_range

# id = int(playerid_lookup('judge', 'aaron')['key_mlbam'])
# speeds = statcast_running_splits(2022, 50)
# print(speeds)
# hitter_stats = statcast_batter('2022-06-01', '2022-06-30', id)
# # hitter_stats.head(2)


def compute_PA_stats(data, n, timeframes):
    
    data_start_dt = datetime.strptime(data['game_date'].iloc[-1], '%Y-%m-%d').date()
    PAs = []
    # Transform data to only contain 1 row for each PA
    # for index, row in data.iterrows():
    #     if not pd.isnull(row['events'])  :
    #         # 'intentionally' in row['des']
    #     # if row['events'] in ['strikeout', 'field_error', 'field_out', 'single', 'double', 'triple', 'home_run', 'double_play', 'grounded_into_double_play', 'force_out', 'sac_fly', 'fielders_choice', 'walk', 'hit_by_pitch', 'triple_play']:
    #         PAs.append(row)
    PA_df = data.dropna(subset = ['events'])
    print(len(PA_df))
    PA_stats = ['walk', 'strikeout', 'hit_by_pitch', 'ground_ball', 'fly_ball', 'line_drive', 'popup', 'xwoba']
    stats_dict = dict(zip(PA_stats, ([] for _ in PA_stats)))
    time_frame_stats = dict(zip(PA_stats, ([] for _ in PA_stats)))
    used_idx = []
    # Select random samples
    for i in range(n):
        print(i)
        idx = random.randint(0, len(PA_df))
        counter = 0
        while (idx in used_idx or datetime.strptime(PA_df['game_date'].iloc[idx], '%Y-%m-%d').date() < data_start_dt + timedelta(days=timeframes[-1])):
            counter +=1 
            idx =  (idx+1) % len(PA_df)
            if counter == len(PA_df):
                return concatenate_PA_data(time_frame_stats, stats_dict, PA_stats) 
            

        used_idx.append(idx)
        id = int(PA_df['batter'].iloc[idx])
 
        # id = int(playerid_lookup(last_nm, first_nm)['key_mlbam'])
        date = datetime.strptime(PA_df['game_date'].iloc[idx], '%Y-%m-%d').date()
        player_data = PA_df[PA_df['batter'].astype(int) == id]
        timeframe_data = []
        for index, row in player_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                timeframe_data.append(row)
        player_timeframe_data = pd.DataFrame(timeframe_data)

       
        stats_dict = find_PA_outcome(stats_dict, PA_df, idx)
        
        for stat in stats_dict:
            time_frame_stats[stat].append(find_PA_time_stats(player_timeframe_data, stat, timeframes, date))
    return concatenate_PA_data(time_frame_stats, stats_dict, PA_stats)
def concatenate_PA_data(time_frame_stats, stats_dict, PA_stats):
    PA_data = dict(zip(PA_stats, ([] for _ in PA_stats)))
    for stat in PA_stats:
        
        PA_stat_data = []
        for i in range (n):
            data_row = time_frame_stats[stat][i]
            data_row.append(stats_dict[stat][i])
            
            PA_stat_data.append(data_row)
        PA_data[stat] = PA_stat_data
    return(PA_data)
def find_PA_outcome(stats_dict, PA_df, idx):
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
            time_frame_stats.append(0)
        else:
            time_frame_stats.append(time_frame_stat_ctr[time]/time_frame_PA_ctr[time])
    return time_frame_stats
def compute_bb_stats(data, n, timeframes):
    bb_df = data.dropna(subset = ['type', 'launch_speed', 'launch_angle'])
    bb_df = bb_df[bb_df['type'] == 'X']
    print(len(bb_df))
    bb_stats = ['launch_speed', 'launch_angle']
    stats_dict = dict(zip(bb_stats, ([] for _ in bb_stats)))
    data_start_dt = datetime.strptime(bb_df['game_date'].iloc[-1], '%Y-%m-%d').date()
    time_frame_stats = dict(zip(bb_stats, ([] for _ in bb_stats)))

    used_idx = []
    for i in range(n):
    
        idx = random.randint(0, len(bb_df))
        counter = 0
        while (idx in used_idx or datetime.strptime(bb_df['game_date'].iloc[idx], '%Y-%m-%d').date() < data_start_dt + timedelta(days=timeframes[-1])):
            counter +=1 
            idx =  (idx+1) % len(bb_df)
            if counter == len(bb_df):
                return concatenate_PA_data(time_frame_stats, stats_dict, bb_stats) 

        used_idx.append(idx)
        id = int(bb_df['batter'].iloc[idx])
        date = datetime.strptime(bb_df['game_date'].iloc[idx], '%Y-%m-%d').date()
        player_data = bb_df[bb_df['batter'].astype(int) == id]
        timeframe_data = []
        for index, row in player_data.iterrows():
            if datetime.strptime(row['game_date'], '%Y-%m-%d').date() < date:
                timeframe_data.append(row)
        player_timeframe_data = pd.DataFrame(timeframe_data)
        
        stats_dict = find_bb_outcome(stats_dict, bb_df, idx)
        for stat in bb_stats:
            time_frame_stats[stat].append(find_bb_time_stats(player_timeframe_data, stat, timeframes, date))
    return concatenate_bb_data(bb_stats, time_frame_stats, stats_dict)
def concatenate_bb_data(bb_stats, time_frame_stats, stats_dict):
    bb_data = dict(zip(bb_stats, ([] for _ in bb_stats)))
    for stat in bb_stats:
        
        bb_stat_data = []
        for i in range (n):
            data_row = time_frame_stats[stat][i]
            data_row.append(stats_dict[stat][i])
            
            bb_stat_data.append(data_row)
        bb_data[stat] = bb_stat_data
    return(bb_data)
def find_bb_outcome(stats_dict, bb_df, idx):
    stats_dict['launch_speed'].append(bb_df['launch_speed'].iloc[idx])
    stats_dict['launch_angle'].append(bb_df['launch_angle'].iloc[idx])
    return stats_dict
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
            time_frame_stats.append(0)
        else:
            time_frame_stats.append(time_frame_stat_ctr[time]/time_frame_PA_ctr[time])
    return time_frame_stats

timeframes = [1, 15, 30, 60]
n = 100
PA_data = compute_PA_stats(data, n, timeframes)
bb_data = compute_bb_stats(data, n, timeframes)
print(bb_data)



#%%


# %%
