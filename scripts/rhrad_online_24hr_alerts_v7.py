# RHR Online Anomaly Detection & Alert Monitoring
######################################################
# Author: Gireesh K. Bogu                            #
# Email: gbogu17@stanford.edu                        #
# Location: Dept.of Genetics, Stanford University    #
# Date: Oct 29 2020                                #
######################################################

# uses raw heart rate and steps data (this stpes data doesn't have zeroes and need to innfer from hr datetime stamp)

## simple command
# python rhrad_online_alerts.py --heart_rate hr.csv --steps steps.csv

## full command
#  python rhrad_online_alerts.py --heart_rate pbb_fitbit_oldProtocol_hr.csv --steps pbb_fitbit_oldProtocol_steps.csv --myphd_id pbb_RHR_online --figure1 pbb_RHR_online_anomalies.pdf --anomalies pbb_RHR_online_anomalies.csv --symptom_date 2020-01-10 --diagnosis_date 2020-01-11 --outliers_fraction 0.1 --random_seed 10  --baseline_window 744 --sliding_window 1 --alerts pbb_RHR_online_alerts.csv --figure2 pbb_RHR_online_alerts.pdf

# python rhrad_online_alerts.py --heart_rate pbb_fitbit_oldProtocol_hr.csv \
# --steps pbb_fitbit_oldProtocol_steps.csv \
# --myphd_id pbb_RHR_online \
# --figure1 pbb_RHR_online_anomalies.pdf \
# --anomalies pbb_RHR_online_anomalies.csv \
# --symptom_date 2020-01-10 --diagnosis_date 2020-01-11 \
# --outliers_fraction 0.1 \
# --random_seed 10  \
# --baseline_window 744 --sliding_window 1 
# --alerts pbb_RHR_online_alerts.csv \
# --figure2 pbb_RHR_online_alerts.pdf

import warnings
warnings.filterwarnings('ignore')
import sys 
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope


####################################

parser = argparse.ArgumentParser(description='Find anomalies in wearables time-series data.')
parser.add_argument('--heart_rate', metavar='', help ='raw heart rate count with a header = heartrate')
parser.add_argument('--steps',metavar='', help ='raw steps count with a header = steps')
parser.add_argument('--myphd_id',metavar='', default = 'myphd_id', help ='user myphd_id')
parser.add_argument('--anomalies', metavar='', default = 'myphd_id_anomalies.csv', help='save predicted anomalies as a CSV file')
parser.add_argument('--figure1', metavar='',  default = 'myphd_id_anomalies.pdf', help='save predicted anomalies as a PDF file')
parser.add_argument('--symptom_date', metavar='', default = 'NaN', help = 'symptom date with y-m-d format')
parser.add_argument('--diagnosis_date', metavar='', default = 'NaN',  help='diagnosis date with y-m-d format')
parser.add_argument('--outliers_fraction', metavar='', type=float, default=0.1, help='fraction of outliers or anomalies')
parser.add_argument('--random_seed', metavar='', type=int, default=10, help='random seed')
parser.add_argument('--baseline_window', metavar='',type=int, default=744, help='baseline window is used for training (in hours)')
parser.add_argument('--sliding_window', metavar='',type=int, default=1, help='sliding window is used to slide the testing process each hour')
parser.add_argument('--alerts', metavar='', default = 'myphd_id_alerts.csv', help='save predicted anomalies as a CSV file')
parser.add_argument('--figure2', metavar='',  default = 'myphd_id_alerts.pdf', help='save predicted anomalies as a PDF file')

args = parser.parse_args()


# as arguments
fitbit_oldProtocol_hr = args.heart_rate
fitbit_oldProtocol_steps = args.steps
myphd_id = args.myphd_id
myphd_id_anomalies = args.anomalies
myphd_id_figure1 = args.figure1
symptom_date = args.symptom_date
diagnosis_date = args.diagnosis_date
RANDOM_SEED = args.random_seed
outliers_fraction =  args.outliers_fraction
baseline_window = args.baseline_window
sliding_window = args.sliding_window
myphd_id_alerts = args.alerts
myphd_id_figure2 = args.figure2


####################################

class RHRAD_online:

    # Infer resting heart rate ------------------------------------------------------

    def resting_heart_rate(self, heartrate, steps):
        """
        This function uses heart rate and steps data to infer resting heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
        """
        # heart rate data
        df_hr = pd.read_csv(fitbit_oldProtocol_hr)
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        # steps data
        df_steps = pd.read_csv(fitbit_oldProtocol_steps)
        df_steps = df_steps.set_index('datetime')
        df_steps.index.name = None
        df_steps.index = pd.to_datetime(df_steps.index)

        # merge dataframes
        #df_hr = df_hr.resample('1min').mean()
        #df_steps = df_steps.resample('1min').mean()

        # added "outer" paramter for merge function to adjust the script to the new steps format
        #df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True, how="outer")
        df1 = df1[pd.isnull(df1).any(axis=1)].fillna(0)
        df1 = df1.rename(columns={"value_x": "heartrate", "value_y": "steps"})

        df1 = df1.resample('1min').mean()
        print(myphd_id)
        print("Data size (in miutes) before removing missing data")
        print(df1.shape)
        ax = df1.plot(figsize=(20,4), title=myphd_id)
        ax.figure.savefig(myphd_id+'_data.png')
        #print(df1)


        df1 = df1.dropna(how='any')
        df1 = df1.loc[df1['heartrate']!=0]
        print("Data size (in miutes) after removing missing data")
        print(df1.shape)
        #print(df1)
        
        # define RHR as the HR measurements recorded when there were less than two steps taken during a rolling time window of the preceding 12 minutes (including the current minute)
        df1['steps'] = df1['steps'].apply(np.int64)
        df1['steps_window_12'] = df1['steps'].rolling(12).sum()
        df1 = df1.loc[(df1['steps_window_12'] == 0 )]
        
        print(df1['heartrate'].describe())
        print(df1['steps_window_12'].describe())

        # impute missing data 
        #df1 = df1.resample('1min').mean()
        #df1 = df1.ffill()

        print("No.of timesteps for RHR (in minutes)")
        print(df1.shape)

        return df1

    # Pre-processing ------------------------------------------------------

    def pre_processing(self, resting_heart_rate):
        """
        This function takes resting heart rate data and applies moving averages to smooth the data and 
        downsamples to one hour by taking the avegare values
        """
        # smooth data
        df_nonas = df1.dropna()
        df1_rom = df_nonas.rolling(400).mean()
        # resample
        df1_resmp = df1_rom.resample('1H').mean()
        df2 = df1_resmp.drop(['steps'], axis=1)
        df2 = df2.dropna()

        print("No.of timesteps for RHR (in hours)")
        print(df2.shape)
        return df2

    # Seasonality correction ------------------------------------------------------

    def seasonality_correction(self, resting_heart_rate, steps):
        """
        This function takes output pre-processing and applies seasonality correction
        """
        sdHR_decomposition = seasonal_decompose(sdHR, model='additive', freq=1)
        sdSteps_decomposition = seasonal_decompose(sdSteps, model='additive', freq=1)
        sdHR_decomp = pd.DataFrame(sdHR_decomposition.resid + sdHR_decomposition.trend)
        sdHR_decomp.rename(columns={sdHR_decomp.columns[0]:'heartrate'}, inplace=True)
        sdSteps_decomp = pd.DataFrame(sdSteps_decomposition.resid + sdSteps_decomposition.trend)
        sdSteps_decomp.rename(columns={sdSteps_decomp.columns[0]:'steps_window_12'}, inplace=True)
        frames = [sdHR_decomp, sdSteps_decomp]
        data = pd.concat(frames, axis=1)
        #print(data)
        #print(data.shape)
        return data

    # Train model and predict anomalies ------------------------------------------------------

    def online_anomaly_detection(self, data_seasnCorec, baseline_window, sliding_window):
        """
        # split the data, standardize the data inside a sliding window 
        # parameters - 1 month baseline window and 1 hour sliding window
        # fit the model and predict the test set

        """
        for i in range(baseline_window, len(data_seasnCorec)):
            data_train_w = data_seasnCorec[i-baseline_window:i] 
            # train data normalization ------------------------------------------------------
            data_train_w += 0.1
            standardizer = StandardScaler().fit(data_train_w.values)
            data_train_scaled = standardizer.transform(data_train_w.values)
            data_train_scaled_features = pd.DataFrame(data_train_scaled, index=data_train_w.index, columns=data_train_w.columns)
            data = pd.DataFrame(data_train_scaled_features)
            data_1 = pd.DataFrame(data).fillna(0)
            data_1['steps'] = '0'
            data_1['steps_window_12'] = (data_1['steps']) 
            data_train_w = data_1
            data_train.append(data_train_w)

            data_test_w = data_seasnCorec[i:i+sliding_window] 
            # test data normalization ------------------------------------------------------
            data_test_w += 0.1
            data_test_scaled = standardizer.transform(data_test_w.values)
            data_scaled_features = pd.DataFrame(data_test_scaled, index=data_test_w.index, columns=data_test_w.columns)
            data = pd.DataFrame(data_scaled_features)
            data_1 = pd.DataFrame(data).fillna(0)
            data_1['steps'] = '0'
            data_1['steps_window_12'] = (data_1['steps']) 
            data_test_w = data_1
            data_test.append(data_test_w)

            # fit the model  ------------------------------------------------------
            model = EllipticEnvelope(random_state=RANDOM_SEED,
                                     support_fraction=0.7,
                                     contamination=outliers_fraction).fit(data_train_w)
            # predict the test set
            preds = model.predict(data_test_w)
            #preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
            dfs.append(preds)


    # Merge predictions ------------------------------------------------------

    def merge_test_results(self, data_test):
        """
        Merge predictions

        """
        # concat all test data (from sliding window) with their datetime index and others
        data_test = pd.concat(data_test)
        # merge predicted anomalies from test data with their corresponding index and other features 
        preds = pd.DataFrame(dfs)
        preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
        data_test_df = pd.DataFrame(data_test)
        data_test_df = data_test_df.reset_index()
        data_test_preds = data_test_df.join(preds)
        return data_test_preds


    # Positive Anomalies -----------------------------------------------------------------
        """
        Selects anomalies in positive direction and saves in a CSV file

        """
    def positive_anomalies(self, data):
        a = data.loc[data['anomaly'] == -1, ('index', 'heartrate')]
        positive_anomalies = a[(a['heartrate']> 0)]
        # Anomaly results
        positive_anomalies['Anomalies'] = myphd_id
        positive_anomalies.columns = ['datetime', 'std.rhr', 'name']
        positive_anomalies.to_csv(myphd_id_anomalies, header=True) 
        return positive_anomalies


    # Alerts  ------------------------------------------------------

    def create_alerts(self, anomalies, data, fitbit_oldProtocol_hr):
        """
        # creates alerts at every 24 hours and send at 9PM.
        # visualise alerts

        """
        # function to assign different alert names
        # summarize hourly alerts
        def alert_types(alert):
            if alert['alerts'] >=6:
                return 'RED'
            elif alert['alerts'] >=1:
                return 'YELLOW'
            else:
                return 'GREEN'

        # summarize hourly alerts
        #anomalies.columns = ['datetime', 'std.rhr', 'name']
        anomalies = anomalies[['datetime']]
        anomalies['datetime'] = pd.to_datetime(anomalies['datetime'], errors='coerce')
        anomalies['alerts'] = 1
        anomalies = anomalies.set_index('datetime')
        anomalies = anomalies[~anomalies.index.duplicated(keep='first')]
        anomalies = anomalies.sort_index()
        alerts = anomalies.groupby(pd.Grouper(freq = '24H',  base=21)).cumsum()
        # apply alert_types function
        alerts['alert_type'] = alerts.apply(alert_types, axis=1)
        alerts_reset = alerts.reset_index()
        #print(alerts_reset)
        # save alerts
        #alerts.to_csv(myphd_id_alerts, mode='a', header=True) 


        # summarize hourly alerts to daily alerts
        daily_alerts = alerts_reset.resample('24H', on='datetime', base=21, label='right').count()
        daily_alerts = daily_alerts.drop(['datetime'], axis=1)
        #print(daily_alerts)

        # function to assign different alert names
        def alert_types(alert):
            if alert['alert_type'] >=6:
                return 'RED'
            elif alert['alert_type'] >=1:
                return 'YELLOW'
            else:
                return 'GREEN'

        # apply alert_types function
        daily_alerts['alert_type'] = daily_alerts.apply(alert_types, axis=1)
        

        # merge missing 'datetime' with 'alerts' as zero aka GREEN
        data1 = data[['index']]
        data1['alert_type'] = 0
        data1 = data1.rename(columns={"index": "datetime"})
        data1['datetime'] = pd.to_datetime(data1['datetime'], errors='coerce')
        data1 = data1.resample('24H', on='datetime', base=21, label='right').count()
        data1 = data1.drop(data1.columns[[0,1]], axis=1)
        data1 = data1.reset_index()
        data1['alert_type'] = 0

        data3 = pd.merge(data1, daily_alerts, on='datetime', how='outer')
        data4 = data3[['datetime', 'alert_type_y']]
        data4 = data4.rename(columns={ "alert_type_y": "alert_type"})
        daily_alerts = data4.fillna("GREEN")
        daily_alerts = daily_alerts.set_index('datetime')
        daily_alerts = daily_alerts.sort_index()
  

        # merge alerts with main data and pass 'NA' when there is a missing day instead of 'GREEN'
        df_hr = pd.read_csv(fitbit_oldProtocol_hr)

        df_hr['datetime'] = pd.to_datetime(df_hr['datetime'], errors='coerce')
        df_hr = df_hr.resample('24H', on='datetime', base=21, label='right').mean()
        df_hr = df_hr.reset_index()
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)
        
        df3 = pd.merge(df_hr, daily_alerts, how='outer', left_index=True, right_index=True)
        df3 = df3[df3.alert_type.notnull()]
        df3.loc[df3.heartrate.isna(), 'alert_type'] = pd.NA


        daily_alerts = df3.drop('heartrate', axis=1)
        daily_alerts = daily_alerts.reset_index()
        daily_alerts = daily_alerts.rename(columns={"index": "datetime"})
        daily_alerts.to_csv(myphd_id_alerts,  na_rep='NA', header=True) 

        
        # visualize hourly alerts
        #colors = {'RED': 'red', 'YELLOW': 'yellow', 'GREEN': ''}
        #ax = alerts['alerts'].plot(kind='bar', color=[colors[i] for i in alerts['alert_type']],figsize=(20,4))
        #ax.set_ylabel('No.of Alerts \n', fontsize = 14) # Y label
        #ax.axvline(pd.to_datetime(symptom_date), color='grey', zorder=1, linestyle='--', marker="v" ) # Symptom date 
        #ax.axvline(pd.to_datetime(diagnosis_date), color='purple',zorder=1, linestyle='--', marker="v") # Diagnosis date
        #plt.xticks(fontsize=4, rotation=90)
        #plt.tight_layout()
        #ax.figure.savefig(myphd_id_figure2, bbox_inches = "tight")
        return daily_alerts


    # Merge alerts  ------------------------------------------------------

    def merge_alerts(self, data_test, alerts):
        """
        Merge  alerts  with their corresponding index and other features 

        """

        data_test = data_test.reset_index()
        data_test['index'] = pd.to_datetime(data_test['index'], errors='coerce')
        test_alerts = alerts
        test_alerts = test_alerts.rename(columns={"datetime": "index"})
        test_alerts['index'] = pd.to_datetime(test_alerts['index'], errors='coerce')
        test_alerts = pd.merge(data_test, test_alerts, how='outer', on='index')
        test_alerts.fillna(0, inplace=True)
        #print(test_alerts)
        return test_alerts

    
    # Visualization and save predictions ------------------------------------------------------

    def visualize(self, results, positive_anomalies, test_alerts, symptom_date, diagnosis_date):
        """
        visualize all the data with anomalies and alerts

        """
        try:

            with plt.style.context('seaborn-dark-palette'):

                fig, ax = plt.subplots(1, figsize=(16,2.5))
               
                ax.bar(test_alerts['index'], test_alerts['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                training = data_train[0]
                training = training.reset_index()
                ax.bar(training['index'], training['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                colors = {0:'', 'RED': 'red', 'YELLOW': 'yellow', 'GREEN': ''}
        
                for i in range(len(test_alerts)):
                    v = colors.get(test_alerts['alert_type'][i])
                    ax.vlines(test_alerts['index'][i], test_alerts['heartrate'].min(), test_alerts['heartrate'].max(),  linestyle='solid',  color=v)
                
                ax.scatter(positive_anomalies['datetime'],positive_anomalies['std.rhr'], color='red', label='Anomaly', s=4)
                
                # 5371428635153486
                #ax.set_xlim([pd.to_datetime('2026-07-28 21:00:00'), pd.to_datetime('2026-08-21 21:00:00')])
                
                # 2563877235467276
                #ax.set_xlim([pd.to_datetime('2024-10-30 21:00:00'), pd.to_datetime('2024-12-31 21:00:00')])

                # 2561881976463395
                #ax.set_xlim([pd.to_datetime('2027-02-15 21:00:00'), pd.to_datetime('2027-04-01 21:00:00')])

                ax.tick_params(axis='both', which='major', color='blue', labelsize=14)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=14)
                #ax.set_title(myphd_id,fontweight="bold", size=12) # Title
                ax.set_ylabel('Std. RHR\n', fontsize = 12) # Y label
                #ax.axvline(pd.to_datetime(symptom_date), color='black', zorder=1, linestyle='--',   lw=6, alpha=0.5) # Symptom date 
                #ax.axvline(pd.to_datetime(diagnosis_date), color='purple',zorder=1, linestyle='--', lw=6, alpha=0.5) # Diagnosis date
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
                ax.grid(zorder=0)
                ax.grid(False)

                plt.xticks(fontsize=8, rotation=90)
                plt.yticks(fontsize=10)

                #plt.setp(ax.get_xticklabels() [0::2], visible=False) 
                #plt.setp(ax.get_xticklabels() [-1::-2], visible=False)
                #ax.axes.get_xaxis().set_visible(False) # set X axis labels false

                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white') 

                figure = fig.savefig(myphd_id_figure1, bbox_inches='tight')                             
                return figure

        except:
            with plt.style.context('seaborn-dark-palette'):

                fig, ax = plt.subplots(1, figsize=(16,2.5))

                ax.bar(test_alerts['index'], test_alerts['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                training = data_train[0]
                training = training.reset_index()
                ax.bar(training['index'], training['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                print(training)

                colors = {0:'', 'RED': 'red', 'YELLOW': 'yellow', 'GREEN': ''}
        
                for i in range(len(test_alerts)):
                    v = colors.get(test_alerts['alert_type'][i])
                    ax.vlines(test_alerts['index'][i], test_alerts['heartrate'].min(), test_alerts['heartrate'].max(),  linestyle='solid',  lw=2, color=v)
 
                ax.scatter(positive_anomalies['datetime'],positive_anomalies['std.rhr'], color='red', label='Anomaly', s=4)

                # 5371428635153486
                #ax.set_xlim([pd.to_datetime('2026-07-28 21:00:00'), pd.to_datetime('2026-08-21 21:00:00')])
                
                # 2563877235467276
                #ax.set_xlim([pd.to_datetime('2024-06-29 21:00:00'), pd.to_datetime('2024-07-16 21:00:00')])

                # 2561881976463395
                #ax.set_xlim([pd.to_datetime('2027-02-15 21:00:00'), pd.to_datetime('2027-04-01 21:00:00')])

                ax.tick_params(axis='both', which='major', color='blue', labelsize=12)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=12)
                ax.set_title(myphd_id,fontweight="bold", size=12) # Title
                ax.set_ylabel('Std. RHR\n', fontsize = 12) # Y label
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
                ax.grid(zorder=0)
                ax.grid(False)
                plt.xticks(fontsize=8, rotation=90)
                plt.yticks(fontsize=10)
                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white')     
                figure = fig.savefig(myphd_id_figure1, bbox_inches='tight')       
                return figure



model = RHRAD_online()

df1 = model.resting_heart_rate(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
df2 = model.pre_processing(df1)
sdHR = df2[['heartrate']]
sdSteps = df2[['steps_window_12']]
data_seasnCorec = model.seasonality_correction(sdHR, sdSteps)
data_seasnCorec += 0.1
dfs = []
data_train = []
data_test = []
model.online_anomaly_detection(data_seasnCorec, baseline_window, sliding_window)
results = model.merge_test_results(data_test)
positive_anomalies = model.positive_anomalies(results)
alerts = model.create_alerts(positive_anomalies, results, fitbit_oldProtocol_hr)
test_alerts = model.merge_alerts(results, alerts)
model.visualize(results, positive_anomalies, test_alerts, symptom_date, diagnosis_date)
print("Finished!") 
