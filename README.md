# AnomalyDetect
Detects anomalous heart rate from smartwatch data.


#### Install required packages

```
pip install -r requirements.txt
```

#### Wearable data (Heart rate, steps and sleep) from FitBit users with a COVID-19 diagnosis

```
Data from 29 users - AnomalyDetect/data/
For full set (31 users): https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip
```

#### Examples

```
# check results folder for the output

python scripts/rhrad_offline.py --heart_rate data/AHYIJDV_hr.csv --steps data/AHYIJDV_steps.csv
python scripts/hrosad_offline.py --heart_rate data/AHYIJDV_hr.csv --steps data/AHYIJDV_steps.csv
```

## Offline Models 

#### RHR-AD (Resting Heart Rate) Anomaly Detector

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
 

#### HROS-AD (Heart Rate Over Steps) Anomaly Detector

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

## Online Models

In progress .......

#### Help
```
python rhrad_offline.py -h
python hrosad_offline.py -h
```
