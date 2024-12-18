import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from datetime import timedelta
import re
import requests

important_cols = ["Activity Name", "Activity Date", "Distance", "Activity Type", "Elapsed Time", "Max Heart Rate", "Relative Effort", 
                      "Moving Time", "Max Speed", "Average Speed", "Elevation Gain", "Elevation Loss", "Elevation Low", "Elevation High",
                      "Max Grade", "Average Grade", "Max Cadence", "Average Cadence", "Average Heart Rate", "Max Watts", "Average Watts",
                      "Calories", "Total Work", "Weighted Average Power", "Power Count", "Grade Adjusted Distance", "Average Elapsed Speed",
                      "Total Steps", "Training Load", "Intensity", "Average Grade Adjusted Pace", "CSV Type"]

activity_types = ['Running', 'Run', 'Corrida', 'Hardloopsessie', 'Course à pied']

marathon_list = ["Marathon", "Maraton", "CIM", "Olympic Trials", "Maratona", "MARATHON", "marathon", "olympic Trials"]

dutch_map = {
    'jan.': 'Jan',
    'feb.': 'Feb',
    'mrt.': 'Mar',
    'apr.': 'Apr',
    'mei': 'May',
    'jun.': 'Jun',
    'jul.': 'Jul',
    'aug.': 'Aug',
    'sep.': 'Sep',
    'okt.': 'Oct',
    'nov.': 'Nov',
    'dec.': 'Dec'
}

french_map = {
    'janv.': 'Jan',
    'févr.': 'Feb',
    'mars': 'Mar',
    'avr.': 'Apr',
    'mai': 'May',
    'juin': 'Jun',
    'juil.': 'Jul',
    'août': 'Aug',
    'sept.': 'Sep',
    'oct.': 'Oct',
    'nov.': 'Nov',
    'déc.': 'Dec'
}

spanish_map = {
    'jan': 'Jan',
    'fev': 'Feb',
    'mar': 'Mar',
    'abr': 'Apr',
    'mai': 'May',
    'jun': 'Jun',
    'jul': 'Jul',
    'ago': 'Aug',
    'set': 'Sep',
    'out': 'Oct',
    'nov': 'Nov',
    'dez': 'Dec'
}

# Converts csv to df and puts into list
def get_data():
  directory_path = "./data_src"
  marathon_training_list = []
  for filename in os.listdir(directory_path):
    path = os.path.join(directory_path, filename)
    if os.path.isdir(path):
      marathons = os.listdir(path)
      for marathon in marathons:
        marathon_path = os.path.join(path, marathon)
        if os.path.isfile(marathon_path) and marathon_path.endswith('.csv'):
          marathon_training_list.append((marathon, pd.read_csv(marathon_path)))
  return marathon_training_list

# helper function for different format activity csv
def activity_type_first(df):
  df = df[["Title", "Date", "Distance", "Activity Type", "Elapsed Time", "Max HR", 
                "Moving Time", "Best Pace", "Avg Pace", "Total Ascent", "Total Descent", "Min Elevation", "Max Elevation",
                "Max Run Cadence", "Avg Run Cadence", "Avg HR", "Calories"]]
  df.columns = ["Activity Name", "Activity Date", "Distance", "Activity Type", "Elapsed Time", "Max Heart Rate",
                "Moving Time", "Max Speed", "Average Speed", "Elevation Gain", "Elevation Loss", "Elevation Low", 
                "Elevation High", "Max Cadence", "Average Cadence", "Average Heart Rate", "Calories"]
  df = df.iloc[::-1].reset_index(drop = True)
  return df

# helper function for standardize_names
def standardize_names_helper(marathon_training_list, standard_columns, i):
  type = 0
  df = marathon_training_list[i][1]
  if i != 0:
    if marathon_training_list[i][1].columns[0] == "Activity Type":
      type = 1
      df = activity_type_first(df)
      missing_columns = set(important_cols) - set(df.columns)
    else:
      df.columns = standard_columns[:len(df.columns)]
      missing_columns = set(standard_columns) - set(df.columns)
    df = df.copy()
    for col in missing_columns:
      df.loc[:, col] = np.nan
  df.loc[:, "CSV Type"] = type
  return df

# returns csvs with same column names in same format containing only runs
def standardize_names(marathon_training_list):
  if marathon_training_list:
    standard_columns = marathon_training_list[0][1].columns
    for i in range(len(marathon_training_list)):
      df = standardize_names_helper(marathon_training_list, standard_columns, i)
      marathon_training_list[i] = (marathon_training_list[i][0], df[important_cols])
  return marathon_training_list

# converts df column in 00:00:00 format to seconds
def convert_to_seconds(time_string):
  if '.' in time_string:
    time_string = time_string[:time_string.find('.')]
  if len(time_string) == len("0:00"):
    time_string = "00:0" + time_string
  elif len(time_string) == len("00:00"):
    time_string = "00:" + time_string
  elif len(time_string) == len("0:00:00"):
    time_string = "0" + time_string
  time_delta = pd.to_timedelta(time_string)
  return time_delta.total_seconds()

# converts dates to english
def convert_to_english(col):
  col = col.copy()
  combined_map = {**dutch_map, **french_map, **spanish_map}
  for i in range(len(col)):
    if isinstance(col.loc[i], str):
      if "à" in col.loc[i]:
        col.loc[i] = col.loc[i].replace('à', '')
      if "de " in col.loc[i]:
        col.loc[i] = col.loc[i].replace('de ', '')
      for month_name, english_month in combined_map.items():
        if month_name in col.loc[i]:
          col.loc[i] = col.loc[i].replace(month_name, english_month)
  return col

# standardizes all other column types
def standardize_others(df, col):
  for k in range(len(df)):
    if isinstance(df.loc[k, col], str):
      if '\xa0' in df.loc[k, col]:
        df.loc[k, col] = df.loc[k, col].replace('\xa0', '')
      if ',' in df.loc[k, col]:
        df.loc[k, col] = df.loc[k, col].replace(',', '.')
      if ':' in df.loc[k, col]:
        if df.loc[k, col] != "00:NaN:NaN":
          df.loc[k, col] = convert_to_seconds(df.loc[k, col])
        else:
          df.loc[k, col] = df.loc[k, "Elapsed Time"]
      elif '--' in df.loc[k, col]:
        df.loc[k, col] = df.loc[k, col].replace('--', '0')
  df.loc[:, col] = df.loc[:, col].astype('float64')
  return df

# makes all corresponding columns same types
def standardize_types(marathon_training_list):
  marathon_training_list = standardize_names(marathon_training_list)
  for j in range(len(marathon_training_list)):
    name, df = marathon_training_list[j]
    df = df.copy()
    for col in df.columns:
      if col == "Activity Type" or col == "Activity Name":
        df[col] = df[col].astype('str')
      elif col == "Activity Date":
        try:
          df.loc[:, col] = pd.to_datetime(df.loc[:, col])
        except:
          df.loc[:, col] = pd.to_datetime(convert_to_english(df.loc[:, col]))
      else:
        df = standardize_others(df, col)
    marathon_training_list[j] = (name, df)
  return marathon_training_list

# standardizes all units to metric system
def standardize_units(marathon_training_list):
  marathon_training_list = standardize_types(marathon_training_list)
  for i in range(len(marathon_training_list)):
    name, df = marathon_training_list[i]
    if df['CSV Type'][0] == 1:
      df['Distance'] = df['Distance'] * 1.60934
      df[['Average Speed', 'Max Speed']] = df[['Average Speed', 'Max Speed']] * 0.621371
      df[["Elevation Gain", "Elevation Loss", "Elevation Low", "Elevation High"]] * 0.3048
    marathon_training_list[i] = (name, df)
  return marathon_training_list

# removes workouts prior to marathon block
def eighteen_week_cutoff(result_df):
  marathon_date = result_df.iloc[-1]["Activity Date"]
  cutoff_date = marathon_date - timedelta(weeks=18)
  result_df = result_df[result_df["Activity Date"] >= cutoff_date]  
  return result_df

# creates test data folder to view current marathon blocks
def create_test_folder():
  folder_name = "TestData"
  if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
  os.makedirs(folder_name)

# splits up data into marathon blocks
def split_blocks(marathon_training_list):
  marathon_training_list = standardize_units(marathon_training_list)
  new_marathon_training_list = []
  # create_test_folder()
  pattern = '|'.join(map(re.escape, marathon_list))
  for (name, df) in marathon_training_list:
    result_dfs = []
    split_indices = df[(df['Distance'] > 39) & (df['Distance'] < 46) & (df['Activity Type'].isin(activity_types)) 
                       & ((df['Activity Name'].str.contains(pattern, case=False, na=False)))].index
    start_index = 0
    for i, index in enumerate(split_indices):
      result_df = eighteen_week_cutoff(df.loc[start_index:index, :])
      result_df[result_df.select_dtypes(include=['object']).columns] = result_df.select_dtypes(include=['object']).fillna("")
      result_df[result_df.select_dtypes(include=['float', 'int']).columns] = result_df.select_dtypes(include=['float', 'int']).fillna(0)
      result_df = result_df.infer_objects(copy=False)
      result_dfs.append(result_df)
      # filename = f"{name}_block_{i + 1}.csv"
      # result_df.to_csv("TestData/" + filename, index = False)
      start_index = index + 1
    new_marathon_training_list.append((name, result_dfs))
  return new_marathon_training_list

# gets number of runs, cross training activities and rest days for each block
def count_run_ct_rest_days(marathon_training_list):
  cross_rest_days = []
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      current_date = df.iloc[0]["Activity Date"].date()
      marathon_date = df.iloc[-1]["Activity Date"].date()
      rest_day_counter = 0
      while current_date <= marathon_date:
        if current_date not in df["Activity Date"].tolist():
          rest_day_counter += 1
        current_date += timedelta(days=1)
      cross_train_rows = df[~df["Activity Type"].isin(activity_types)]
      run_rows = df[df["Activity Type"].isin(activity_types)]
      cross_train_counter = len(cross_train_rows)
      run_counter = len(run_rows)
      cross_rest_days.append({
                              "# Cross Training Sessions": cross_train_counter,
                              "# Rest Days": rest_day_counter,
                              "# Runs": run_counter})
  return cross_rest_days

# finds weekly values for each week of block
def calculate_weekly_values(df, marathon_date, current_date):
  week_counter = 0
  weekly_cross_train_distance = 0
  weekly_cross_train_time = 0
  weekly_run_distance = 0
  weekly_run_time = 0
  while current_date <= marathon_date:
    next_week_date = current_date + timedelta(days = 7)
    weekly_data = df[(df["Activity Date"] >= pd.Timestamp(current_date)) & 
                    (df["Activity Date"] < pd.Timestamp(next_week_date))]
    weekly_cross_train_distance += weekly_data[~weekly_data["Activity Type"].isin(activity_types)]["Distance"].sum()
    weekly_cross_train_time += weekly_data[~weekly_data["Activity Type"].isin(activity_types)]["Elapsed Time"].sum()
    weekly_run_distance += weekly_data[weekly_data["Activity Type"].isin(activity_types)]["Distance"].sum()
    weekly_run_time += weekly_data[weekly_data["Activity Type"].isin(activity_types)]["Elapsed Time"].sum()
    current_date = next_week_date
    week_counter += 1
  if week_counter == 0:
    week_counter = 1
  return (week_counter, weekly_cross_train_time / week_counter, weekly_cross_train_distance / week_counter, 
          weekly_run_time / week_counter, weekly_run_distance / week_counter, week_counter * weekly_run_time)

# calculates weekly average distance and time spent cross training
def calculate_weekly_distance_time(marathon_training_list):
  weekly_values = []
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      current_date = df.iloc[0]["Activity Date"].date()
      marathon_date = df.iloc[-1]["Activity Date"].date()
      weekly_value = calculate_weekly_values(df, marathon_date, current_date)
      weekly_values.append({
                            "Number of Training Weeks": weekly_value[0],
                            "Average Weekly Cross Training Time": weekly_value[1],
                            "Average Weekly Cross Training Distance": weekly_value[2],
                            "Average Weekly Running Time": weekly_value[3],
                            "Average Weekly Running Distance": weekly_value[4],
                            "Total Running Time": weekly_value[5]})
  return weekly_values

# finds longest long run and returns run stats
def find_long_run(marathon_training_list):
  longest_runs = []
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      df = df[df["Activity Type"].isin(activity_types)]
      df_without_marathon = df.iloc[:-1]  # Exclude the last row
      max_run = df_without_marathon[df_without_marathon["Distance"] == df_without_marathon["Distance"].max()].iloc[0]
      longest_runs.append(max_run)
  return longest_runs

# finds conditions for the day of the marathon
def get_weather_data(city_name, date):
  date = date.strftime("%Y-%m-%d %H:%M:%S")
  date_only = date.split(' ')[0]
  API_KEY = '8f3ef6bc690b5cb7e20d4e5589200dad'
  url = f'http://api.weatherstack.com/historical'
  params = {
      'access_key': API_KEY,
      'query': city_name,
      'historical_date': date_only,
  }
  response = requests.get(url, params=params)
  data = response.json()
  if response.status_code == 200 and 'error' not in data:
    historical_data = data.get('historical', {}).get(date_only, {})
    if historical_data:
      min_temp = historical_data.get('mintemp')
      max_temp = historical_data.get('maxtemp')
      avg_temp = historical_data.get('avgtemp')
      uv_index = historical_data.get('uv_index')
  return min_temp, max_temp, avg_temp, uv_index

def extract_city_name(activity_name):
  pattern = r'\b(?:' + '|'.join(marathon_list) + r')\b'
  match = re.search(pattern, activity_name)
  if match:
    city_name = activity_name[:match.start()].strip()
    return city_name
  else:
    return None

def scrape_weather_condition(marathon_training_list):
  weather_conditions = []
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      df = df.copy()
      activity_name = df.iloc[-1]["Activity Name"]
      city_name = extract_city_name(activity_name)
      activity_date = df.iloc[-1]["Activity Date"]
      weather_conditions.append(get_weather_data(city_name, activity_date))
  return weather_conditions

def get_marathon_names(marathon_training_list):
  marathon_names = []
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      df = df.copy()
      activity_name = df.iloc[-1]["Activity Name"]
      marathon_names.append(activity_name)
  return marathon_names

def create_dataframe(marathon_training_list, run_ct_rest_days, weekly_values, long_run_data, weather_condition_data, marathon_names):
  final_data = []
  i = 0
  for (name, result_dfs) in marathon_training_list:
    for df in result_dfs:
      final_data.append({
        "Training Data": df,
        "Number of Runs, Rest Days, Cross Training": run_ct_rest_days[i],
        "Weekly Values": weekly_values[i],
        "Long Run Data": long_run_data[i],
        "Weather Conditions": weather_condition_data[i],
        "Marathon Name": marathon_names[i],
        "MarathonTime": df.iloc[-1]["Elapsed Time"]
      })
      i += 1
  marathon_df = pd.DataFrame(final_data)
  return marathon_df

def create_features(marathon_training_list):
  marathon_training_list = split_blocks(marathon_training_list)
  run_ct_rest_days = count_run_ct_rest_days(marathon_training_list)
  weekly_values = calculate_weekly_distance_time(marathon_training_list)
  long_run_data = find_long_run(marathon_training_list)
  weather_condition_data = scrape_weather_condition(marathon_training_list)
  marathon_names = get_marathon_names(marathon_training_list)
  marathon_df =  create_dataframe(marathon_training_list, run_ct_rest_days, weekly_values, 
                                  long_run_data, weather_condition_data, marathon_names)
  return marathon_df

def plot_histogram():
  df = pd.read_csv('marathon_training_data.csv')
  df["MarathonTimeMinutes"] = df["MarathonTime"] / 60

  adjusted_times = df["MarathonTimeMinutes"] - 120
  adjusted_times = adjusted_times[adjusted_times >= 0]  # Exclude times below 2 hours

  bins = list(range(0, 24 * 10, 10))  # 10-minute increments, 23 intervals (0 to 230 minutes)

  plt.figure(figsize=(10, 6))
  plt.hist(adjusted_times, bins=bins, edgecolor='black', alpha=0.7)

  labels = [str(i) for i in range(0, 24)]  # Labels from 0 to 23
  plt.xticks(bins, labels, rotation=45)

  plt.title("Distribution of Marathon Completion Times (10-Minute Intervals Starting at 2 Hours)")
  plt.xlabel("Time Interval (10-Minute Steps from 2 Hours)")
  plt.ylabel("Frequency")
  plt.grid(axis='y', linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.show()

def main():
  marathon_df = create_features(get_data())
  marathon_df.to_csv('marathon_training_data.csv', index=False)
  # plot_histogram()

  
if __name__ == "__main__":
  main()