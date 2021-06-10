# AnomalyDetect
The current version of `AnomalyDetect` detects anomalous heart rate from smartwatch data. `AnomalyDetect` can be used with either RHR (Resting Heart Rate) or HROS (Heart Rate Over Steps) as an input. This method can be applied to any smartwatch data like FitBit, Apple, Garmin and Empatica.


#### Install required packages

```
pip install -r requirements.txt
```


## Offline Models 

Offline models use all the data to find anomalies.

#### RHR-AD (Resting Heart Rate) Offline Anomaly Detector

Full command 
```
python rhrad_offline.py \
       --heart_rate hr.csv \
       --steps steps.csv \
       --myphd_id id_offline \
       --figure id_offine.pdf \
       --anomalies id_offline_anomalies.csv \
       --symptom_date 2020-01-30 \
       --diagnosis_date 2020-01-31 \
       --outliers_fraction 0.1 \
       --random_seed 10 
 ```
 

#### HROS-AD (Heart Rate Over Steps) Offline Anomaly Detector

Full command 
```
python hrosad_offline.py \
       --heart_rate hr.csv \
       --steps steps.csv \
       --myphd_id id_offline \
       --figure id_offine.pdf \
       --anomalies id_offline_anomalies.csv \
       --symptom_date 2020-01-30 \
       --diagnosis_date 2020-01-31 \
       --outliers_fraction 0.1 \
       --random_seed 10 
 ```
 
 

## Online Model (RHRAD)

It uses RHR data and splits it into training data by taking the first 744 hours as a baseline (1 month) and test data by taking the next 1 hour data, and uses a 1 hour sliding window to find anomalies in the test data in “real-time”. If the anomalies occur frequently within 24 hours, it will automatically create either warning (yellow) or serious (red) alerts at every 9 P.M. Red alerts were set if the anomalies occurred continuously for more than 5 hours within each 24 hours period and yellow alerts were set if the anomalies occurred for one or continuously for less than 5 hours and green alerts were set if there were no anomalies. 

Note: Uses a new Fitbit data format compared to offline models.

Full command
```
python rhrad_online_alerts.py --heart_rate fitbit_newProtocol_hr.csv \
       --steps pbb_fitbit_newProtocol_steps.csv \
       --myphd_id id_RHR_online \
       --figure1 id_RHR_online_anomalies.pdf \
       --anomalies id_RHR_online_anomalies.csv \
       --symptom_date 2020-01-10 --diagnosis_date 2020-01-11 \
       --outliers_fraction 0.1 \
       --random_seed 10  \
       --baseline_window 744 \
       --sliding_window 1 \
       --alerts id_RHR_online_alerts.csv \
       --figure2 id_RHR_online_alerts.pdf
```
