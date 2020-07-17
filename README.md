# AnomalyDetect
Detects anomalous heart rate from smartwatch data.


#### Install required packages

```
pip install -r requirements.txt
```

### RHR-AD (Resting Heart Rate) Anomaly Detector

Command 
```
python rhrad_offline.py \
       --heart_rate hr.csv \
       --steps hr.csv \
       --myphd_id id_offline \
       --figure id_offine.pdf \
       --anomalies id_offline_anomalies.csv \
       --symptom_date 2020-01-30 \
       --diagnosis_date 2020-01-31 \
       --outliers_fraction 0.1 \
       --random_seed 10 
 ```
 

### HROS-AD (Heart Rate Over Steps) Anomaly Detector

Command
```
python hrosad_offline.py \
       --heart_rate hr.csv \
       --steps hr.csv \
       --myphd_id id_offline \
       --figure id_offine.pdf \
       --anomalies id_offline_anomalies.csv \
       --symptom_date 2020-01-30 \
       --diagnosis_date 2020-01-31 \
       --outliers_fraction 0.1 \
       --random_seed 10 
 ```
