### --------------------------------------------------------------------------
### PART 1 - Configuration
### --------------------------------------------------------------------------
# @jbuddavis
# https://github.com/jbuddavis

# Load modules
import pandas as pd 
import time
import cfbd
from sklearn import linear_model

# track time to complete, turn off pandas warning
start = time.time()
pd.options.mode.chained_assignment = None

#%% Configure Inputs
# Choose what year you would like to perform adjustment on
year = 2019 # year of interest

# Configure API key authorization
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = '+txX/9qrny69W7psXOiRTZmVDPE+vObxHmJRYdo3r0lJFk3lg5UyUWFOKgCizCrT'
configuration.api_key_prefix['Authorization'] = 'Bearer'

# create empty dataframes to be filled
dfCal = pd.DataFrame() # dataframe for calendar
dfPBP = pd.DataFrame() # dataframe for pbp data
dfGame = pd.DataFrame() # dataframe for game information
dfTeam = pd.DataFrame() # dataframe for team information

### --------------------------------------------------------------------------
### PART 2 - Ping the API
### --------------------------------------------------------------------------
# get calendar for season
api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
api_response = api_instance.get_calendar(year)
dfCal = pd.DataFrame().from_records([g.to_dict()for g in api_response])

# loop through the calendar and get the PBP for each week
print('Getting PBP data for year '+str(year)+'...')
for i in range (0,len(dfCal)):
    # iterate through calendar to get week/season_type variables to pass to the API
    week = int(dfCal.loc[i,'week']) # get week from calendar
    season_type = dfCal.loc[i,'season_type'] # get season type from calendar
    # Get play-by-play Data
    api_instance = cfbd.PlaysApi(cfbd.ApiClient(configuration))
    api_response = api_instance.get_plays(year, week, season_type=season_type)
    dfWk = pd.DataFrame().from_records([g.to_dict()for g in api_response])
    dfPBP = dfPBP.append(dfWk)
    # Get game info (used for homefield advantage)
    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    api_response = api_instance.get_games(year, week=week, season_type=season_type)
    dfGameWk = pd.DataFrame().from_records([g.to_dict()for g in api_response])
    dfGame = dfGame.append(dfGameWk)
    print('PBP data downloaded for '+season_type+' week',week)

# Get FBS teams
api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))
api_response = api_instance.get_fbs_teams(year=year) 
dfTeam = pd.DataFrame().from_records([g.to_dict()for g in api_response])
dfTeam = dfTeam[['school']]
dfTeam.to_csv('teams.csv',index=False) #print FBS teams to csv for record keeping

### --------------------------------------------------------------------------
### PART 3 - Format the data
### --------------------------------------------------------------------------
print('Formatting data...')

# Drop non-"fbs-vs-fbs" games
dfPBP = dfPBP[dfPBP['home'].isin(dfTeam.school.to_list())] # drop data if home team not in list of teams
dfPBP = dfPBP[dfPBP['away'].isin(dfTeam.school.to_list())] # drop data if away team not in list of teams
dfPBP.reset_index(inplace=True,drop=True)
dfGame = dfGame[dfGame['home_team'].isin(dfTeam.school.to_list())]
dfGame = dfGame[dfGame['away_team'].isin(dfTeam.school.to_list())]
dfGame.to_csv('games'+str(year)+'.csv', index=False) # print game data to csv for record keeping
dfGame.reset_index(inplace=True,drop=True)   

# drop nas
dfPBP.dropna(subset=['ppa'],inplace=True)
dfPBP.reset_index(inplace=True,drop=True)

# create list of neutral site games
neutralGames = dfGame['id'][dfGame['neutral_site']==True].to_list()

# All Plays
df = dfPBP[['game_id','home','offense','defense','ppa']] # columns of interest
df.loc['hfa'] = None # homefield advantage
df.loc[(df.home == df.offense),'hfa']=1 # home team on offense
df.loc[(df.home == df.defense),'hfa']=-1 # away team on offense
df.loc[(df.game_id.isin(neutralGames)),'hfa']=0 # neutral site games
df = df[['offense','hfa','defense','ppa']] # drop unneeded colums
df.dropna(subset=['ppa'],inplace=True) # drop nas
df.reset_index(inplace=True,drop=True) # reset index
df.to_csv('allPBP'+str(year)+'.csv',index=False) # output to csv for record keeping

print('Data formatted')

### --------------------------------------------------------------------------
### PART 4 - Initialize Opponent Adjustment
### --------------------------------------------------------------------------
#%% Call the opponent adjustment on our dataframes of interest
# if you just need to tweek the opp-adj func, to save time, after the first 
# round of pbp downloading, comment out lines ~85-167 and read in the pbp csvs here
dfTeam = pd.read_csv('teams.csv')
df = pd.read_csv('allPBP'+str(year)+'.csv')

# dataframe column names to help guide opponent adjustment function
offStr = 'offense' # Column of interest, either the team/player we want to adjust
hfaStr = 'hfa' # Homefield Advantage column name
defStr = 'defense' # Opponent column name
stat = 'ppa' # stat to adjust on

### --------------------------------------------------------------------------
### PART 5 - Perform Opponent Adjustment 
### --------------------------------------------------------------------------
# loop through our list of dataframes & adjust each for opponent & homefield advantage
print('Performing Opponent-Adjustment...')

# Create dummy variables for each Team/Opponent, plus Homefield Advantage
dfDummies = pd.get_dummies(df[[offStr, hfaStr, defStr]])

# Hyperparameter tuning for alpha (aka lambda, ie the penalty term) 
# for full season PBP data, the alpha will be 150-200, for smaller sample sizes it may find a higher alpha
rdcv = linear_model.RidgeCV(alphas = [75,100,125,150,175,200,225,250,275,300,325], fit_intercept = True)
rdcv.fit(dfDummies,df[stat]);
alf = rdcv.alpha_

# Or set Alpha directly here
# alf = 175

# Set up ridge regression model parameters
reg = linear_model.Ridge(alpha = alf, fit_intercept = True)  

# Run the regression
# X values in the regression will be dummy variables each Offense/Defense, plus Homefield Advantage
# y values will be the raw value from each game for the specific stat we're adjusting
reg.fit(X = dfDummies, y = df[stat])

# Extract regression coefficients
dfRegResults = pd.DataFrame({
    'coef_name': dfDummies.columns.values,
    'ridge_reg_coef': reg.coef_})

# Add intercept back in to reg coef to get 'adjusted' value
dfRegResults['ridge_reg_value'] = (dfRegResults['ridge_reg_coef']+reg.intercept_)

#Print the HFA and Alpha values
print('Homefield Advantage: (alpha: '+str(alf)+')')
print('{:.3f}'.format(dfRegResults[dfRegResults['coef_name'] == hfaStr]['ridge_reg_coef'][0]))

# Offense
dfAdjOff = (dfRegResults[dfRegResults['coef_name'].str.slice(0, len(offStr)) == offStr].
   rename(columns = {"ridge_reg_value": stat}).
  reset_index(drop = True))
dfAdjOff['coef_name'] = dfAdjOff['coef_name'].str.replace(offStr+'_','')
dfAdjOff = dfAdjOff.drop(columns=['ridge_reg_coef'])

# Defense
dfAdjDef = (dfRegResults[dfRegResults['coef_name'].str.slice(0, len(defStr)) == defStr].
   rename(columns = {"ridge_reg_value": stat}).
  reset_index(drop = True))
dfAdjDef['coef_name'] = dfAdjDef['coef_name'].str.replace(defStr+'_','')
dfAdjDef = dfAdjDef.drop(columns=['ridge_reg_coef'])

### --------------------------------------------------------------------------
### PART 6 - Join Data and Output
### --------------------------------------------------------------------------

# associate the raw and adjusted epa with each team
dfTeam['rawOff'] = dfTeam.join(df.groupby('offense').mean().ppa, on='school').ppa # raw avg ppa
dfTeam['adjOff'] = dfTeam.join(dfAdjOff.set_index('coef_name'), on='school').ppa # adj est ppa
dfTeam['rawDef'] = dfTeam.join(df.groupby('defense').mean().ppa, on='school').ppa
dfTeam['adjDef'] = dfTeam.join(dfAdjDef.set_index('coef_name'), on='school').ppa

# final formatting and output
dfTeam = dfTeam.round(3) # round adjusted value to thousandths 
print(dfTeam)
dfTeam.to_csv('adj'+str(year)+'.csv', index=False)
print('Adjusted Data ouput to: adj'+str(year)+'.csv')
end = time.time()
print('Time Elapsed (s): ',round(end-start,1))