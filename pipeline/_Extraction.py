import amsterdamumcdb
import pandas as pd
import pandas_gbq
from google.oauth2 import service_account
from utils import ROOT

class Mixin:
    def extract_data(self):
        project_id, config_gbq = set_gbq_settings()
        if self.source == "amsterdamumcdb":
            get_steroids(project_id=project_id, config_gbq=config_gbq)
            get_sepsis(config_gbq=config_gbq) # to be placed after get_* with project_id=project_id
            get_mechanical_ventilation(config_gbq=config_gbq)
            get_features(project_id=project_id, config_gbq=config_gbq)
            get_vasopressors(project_id=project_id, config_gbq=config_gbq)
            get_pf_ratio(project_id=project_id, config_gbq=config_gbq)
            get_arterial_ph(project_id=project_id, config_gbq=config_gbq)
            get_covariate_framework(project_id=project_id, config_gbq=config_gbq)

            print("data was written to csvs")

        if self.source == "mimic":
            get_sepsis_mimic(project_id=project_id, config_gbq=config_gbq)
            get_mortality_mimic(project_id=project_id, config_gbq=config_gbq)
            get_crp_mimic(project_id=project_id, config_gbq=config_gbq)
            get_steroids_mimic(project_id=project_id, config_gbq=config_gbq)
            get_max_gamma_mimic(project_id=project_id, config_gbq=config_gbq)
            get_blood_gas_mimic(project_id=project_id, config_gbq=config_gbq)
            get_features_mimic(project_id=project_id, config_gbq=config_gbq)
            get_pf_ratio(project_id=project_id, config_gbq=config_gbq)
            get_demographics_mimic(project_id=project_id, config_gbq=config_gbq)
            get_ventilation_mimic(project_id=project_id, config_gbq=config_gbq)

            print("data was written to csv")

def set_gbq_settings():
    # set pandas-gbq configurations
    project_id = 'sr-icu-datasets'
    pandas_gbq.context.project = project_id
    config_gbq = {'query': 
            {'defaultDataset': {
                "datasetId": 'ams102', 
                "projectId": 'amsterdamumcdb-data'
                },
            'Location': 'eu'}
            }
    
    return project_id, config_gbq

def set_gbq_settings_mimic(): # TODO TO DELETE, not necessary
    project_id = 'sr-icu-datasets'
    pandas_gbq.context.project = project_id
    config_gbq = {'query': 
            {'defaultDataset': {
                "datasetId": 'mimiciv_derived', 
                "projectId": 'physionet-data'
                },
            'Location': 'eu'}
            }

def get_steroids(project_id, config_gbq):
    # get steroids -> note, get_sepsis_patients function needs to be applied after a generic read_gbq(sql, project_id=project_id, configuration=config_gbq) function for proper authentication with BigQuery
    sql_steroids = """
    SELECT
        d.admissionid,
        d.itemid,
        d.item,
        d.dose,
        d.administered,
        d.administeredunit,
        d.duration,
        (d.start - a.admittedat) AS starttime,
        (d.stop - a.admittedat) AS stoptime
    FROM drugitems d
        LEFT JOIN admissions a
        ON a.admissionid = d.admissionid
    WHERE itemid IN(
        7106, -- Hydrocortison (Solu Cortef)
        6995, -- Dexamethason
        8132, -- Methylprednisolon (Solu-Medrol)
        6922 -- Prednisolon (Prednison)
        )
    AND (d.start - a.admittedat) < (72 * 60 * 60 * 1000) -- start within 72 hours of admission with steroids
    """

    steroids = pd.read_gbq(sql_steroids, project_id=project_id, configuration=config_gbq)
    steroids.to_csv(ROOT + "\\saved_csvs\\steroids.csv")

def get_sepsis(config_gbq):
    # get sepsis patients
    sepsis = amsterdamumcdb.get_sepsis_patients(con=config_gbq)
    sepsis.to_csv(ROOT + "\\saved_csvs\\sepsis.csv")  

    # TODO get covariates: use 24 hours (same as MIMIC-IV)

def get_mechanical_ventilation(config_gbq):
    # get mechanical ventilation
    sql_mechanical_ventilation = """
    SELECT
        l.admissionid,
    CASE
        WHEN COUNT(*) > 0 THEN TRUE
        ELSE FALSE
    END AS mechanical_ventilation_bool,
    STRING_AGG(DISTINCT value, '; ') AS mechanical_ventilation_modes
    FROM listitems l
        LEFT JOIN admissions a 
            ON l.admissionid = a.admissionid
    WHERE
        (
            itemid = 9534  --Type beademing Evita 1
            AND valueid IN (
                1, --IPPV
                2, --IPPV_Assist
                3, --CPPV
                4, --CPPV_Assist
                5, --SIMV
                6, --SIMV_ASB
                7, --ASB
                8, --CPAP
                9, --CPAP_ASB
                10, --MMV
                11, --MMV_ASB
                12, --BIPAP
                13 --Pressure Controled
            )
        )
        OR (
            itemid = 6685 --Type Beademing Evita 4
            AND valueid IN (
                1, --CPPV
                3, --ASB
                5, --CPPV/ASSIST
                6, --SIMV/ASB
                8, --IPPV
                9, --IPPV/ASSIST
                10, --CPAP
                11, --CPAP/ASB
                12, --MMV
                13, --MMV/ASB
                14, --BIPAP
                20, --BIPAP-SIMV/ASB
                22 --BIPAP/ASB
            )
        )
        OR (
            itemid = 8189 --Toedieningsweg O2
            AND valueid = 16 --CPAP
        )
        OR (
            itemid IN (
                12290, --Ventilatie Mode (Set) - Servo-I and Servo-U ventilators
                12347 --Ventilatie Mode (Set) (2) Servo-I and Servo-U ventilators
            )
            AND valueid IN (
                --IGNORE: 1, --Stand By
                2, --PC
                3, --VC
                4, --PRVC
                5, --VS
                6, --SIMV(VC)+PS
                7, --SIMV(PC)+PS
                8, --PS/CPAP
                9, --Bi Vente
                10, --PC (No trig)
                11, --VC (No trig)
                12, --PRVC (No trig)
                13, --PS/CPAP (trig)
                14, --VC (trig)
                15, --PRVC (trig)
                16, --PC in NIV
                17, --PS/CPAP in NIV
                18 --NAVA
            )
        )
        OR (
            itemid = 12376 --Mode (Bipap Vision)
            AND valueid IN (
            1, --CPAP
            2 --BIPAP
        )
        )
        AND (l.measuredat - a.admittedat) < (24 * 60 * 60 * 1000) -- measured within 24 hours after admission
    GROUP BY l.admissionid
    """

    mechanical_ventilation = pd.read_gbq(sql_mechanical_ventilation, configuration=config_gbq)
    mechanical_ventilation.to_csv(ROOT + "\\saved_csvs\\mechanical_ventilation.csv")

def get_features(project_id, config_gbq):
    sql_features = """
    WITH 
    
    numericitems_filtered AS(
    SELECT
        n.admissionid,
        n.itemid,
        n.item,
        n.value,
        n.measuredat,
        n.registeredby,
        CASE
            WHEN NOT registeredby IS NULL THEN TRUE
            ELSE FALSE
        END AS validated
    FROM numericitems n
        LEFT JOIN admissions a 
            ON n.admissionid = a.admissionid
    WHERE n.itemid IN (
        6840, --Natrium
        9555, --Natrium Astrup
        9924, --Natrium (bloed)
        10284, --Na (onv.ISE) (bloed)
        6835, --Kalium mmol/l
        9556, --Kalium Astrup mmol/l
        9927, --Kalium (bloed) mmol/l
        10285, --K (onv.ISE) (bloed) mmol/l
        6836, --Kreatinine µmol/l (erroneously documented as µmol)
        9941, --Kreatinine (bloed) µmol/l
        14216, --KREAT enzym. (bloed) µmol/l
        6779, --Leucocyten 10^9/l
        9965, --Leuco's (bloed) 10^9/l
        9964, --Thrombo's (bloed)
        6797, --Thrombocyten
        10409, --Thrombo's citr. bloed (bloed)
        14252, --Thrombo CD61 (bloed)
        8658, --Temp Bloed
        8659, --Temperatuur Perifeer 2
        8662, --Temperatuur Perifeer 1
        13058, --Temp Rectaal
        13059, --Temp Lies
        13060, --Temp Axillair
        13061, --Temp Oraal
        13062, --Temp Oor
        13063, --Temp Huid
        13952, --Temp Blaas
        16110, --Temp Oesophagus
        9557, --Glucose Astrup
        6833, --Glucose Bloed
        9947, --Glucose (bloed)
        6850, --Ureum
        9943, --Ureum (bloed)
        6825, -- CRP
        10079, -- CRP (bloed)
        6810, -- HCO3
        9992, -- Act.HCO3 (bloed)
        9993, -- act.HCO3 (overig)
        6640, --Hartfrequentie
        10053, --Lactaat (bloed)
        6837, --Laktaat
        9580 --Laktaat Astrup
        --6848, --PH -> separate query to get arterial specimen right
        --12310 --pH (bloed)
    )
    -- measurements within 72 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    AND (measuredat - a.admittedat) <= (1000*60*60*24) --measurements within 24 hours
    ),

    numericitems_unvalidated_filtered AS(
    SELECT
        n.admissionid,
        n.itemid,
        n.item,
        n.value,
        n.measuredat,
        n.registeredby,
        CASE
            WHEN NOT registeredby IS NULL THEN TRUE
            ELSE FALSE
        END AS validated
    FROM numericitems_unvalidated n
        LEFT JOIN admissions a 
            ON n.admissionid = a.admissionid
    WHERE n.itemid IN (
        6840, --Natrium
        9555, --Natrium Astrup
        9924, --Natrium (bloed)
        10284, --Na (onv.ISE) (bloed)
        6835, --Kalium mmol/l
        9556, --Kalium Astrup mmol/l
        9927, --Kalium (bloed) mmol/l
        10285, --K (onv.ISE) (bloed) mmol/l
        6836, --Kreatinine µmol/l (erroneously documented as µmol)
        9941, --Kreatinine (bloed) µmol/l
        14216, --KREAT enzym. (bloed) µmol/l
        6779, --Leucocyten 10^9/l
        9965, --Leuco's (bloed) 10^9/l
        9964, --Thrombo's (bloed)
        6797, --Thrombocyten
        10409, --Thrombo's citr. bloed (bloed)
        14252, --Thrombo CD61 (bloed)
        8658, --Temp Bloed
        8659, --Temperatuur Perifeer 2
        8662, --Temperatuur Perifeer 1
        13058, --Temp Rectaal
        13059, --Temp Lies
        13060, --Temp Axillair
        13061, --Temp Oraal
        13062, --Temp Oor
        13063, --Temp Huid
        13952, --Temp Blaas
        16110, --Temp Oesophagus
        9557, --Glucose Astrup
        6833, --Glucose Bloed
        9947, --Glucose (bloed)
        6850, --Ureum
        9943, --Ureum (bloed)
        6825, -- CRP
        10079, -- CRP (bloed)
        6810, -- HCO3
        9992, -- Act.HCO3 (bloed)
        9993, -- act.HCO3 (overig)
        6640, --Hartfrequentie
        10053, --Lactaat (bloed)
        6837, --Laktaat
        9580 --Laktaat Astrup
        --6848, --PH -> separate query to get arterial specimen right
        --12310 --pH (bloed)
    )
    -- measurements within 72 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    AND (measuredat - a.admittedat) <= (1000*60*60*24) --measurements within 24 hours
    ),
    
    numericitems_complete AS(
    SELECT * 
    FROM numericitems_filtered
    UNION ALL
    SELECT * FROM numericitems_unvalidated_filtered
    )

    SELECT * FROM numericitems_complete
    """

    features = pd.read_gbq(sql_features, project_id=project_id, configuration=config_gbq)
    features.to_csv(ROOT + "\\saved_csvs\\features.csv")

def get_vasopressors(project_id, config_gbq):
    # get vasopressors
    # TODO find out if IS NULL and is NOT NULL are fine instead of is B'0' and B'1' -> it is not. preprocess this part in python
    sql_vaso = """
    WITH dosing AS (
        SELECT  
            drugitems.admissionid, 
            itemid,
            item,
            (start - admissions.admittedat)/(1000*60) AS start_time, 
            (stop - admissions.admittedat)/(1000*60) AS stop_time, 
            duration,
            rate,
            rateunit,
            dose,
            doseunit,
            doseunitid,
            doserateperkg,
            doserateunitid,
            doserateunit,
            CASE
                WHEN weightgroup LIKE '59' THEN 55
                WHEN weightgroup LIKE '60' THEN 65
                WHEN weightgroup LIKE '70' THEN 75
                WHEN weightgroup LIKE '80' THEN 85
                WHEN weightgroup LIKE '90' THEN 95
                WHEN weightgroup LIKE '100' THEN 105
                WHEN weightgroup LIKE '110' THEN 115
                ELSE 80 --mean weight for all years
            END as patientweight
        FROM drugitems 
        LEFT JOIN admissions
        ON drugitems.admissionid = admissions.admissionid
        WHERE ordercategoryid = 65 -- continuous i.v. perfusor
        AND itemid IN (
                7179, -- Dopamine (Inotropin)
                7178, -- Dobutamine (Dobutrex)
                6818, -- Adrenaline (Epinefrine)
                7229  -- Noradrenaline (Norepinefrine)
            )
        AND rate > 0.1
    )
    SELECT 
        admissionid,
        itemid,
        item,
        duration,
        rate,
        rateunit,
        start_time,
        stop_time,
        CASE 
        --recalculate the dose to µg/kg/min ('gamma')
        WHEN doserateperkg = 0 AND doseunitid = 11 AND doserateunitid = 4 --unit: µg/min -> µg/kg/min
            THEN CASE 
                WHEN patientweight > 0
                THEN dose/patientweight
                ELSE dose/80 --mean weight
            END
        WHEN doserateperkg = 0 AND doseunitid = 10 AND
        doserateunitid = 4 --unit: mg/min  -> µg/kg/min
            THEN CASE 
                WHEN patientweight > 0
                THEN dose*1000/patientweight
                ELSE dose*1000/80 --mean weight
            END
        WHEN doserateperkg = 0 AND doseunitid = 10 AND doserateunitid = 5 --unit: mg/uur  -> µg/kg/min
            THEN CASE
                WHEN patientweight > 0
                THEN dose*1000/patientweight/60
                ELSE dose*1000/80 --mean weight
            END
        WHEN doserateperkg = 1 AND doseunitid = 11 AND doserateunitid = 4 --unit: µg/kg/min (no conversion needed)
            THEN dose
        WHEN doserateperkg = 1 AND doseunitid = 11 AND doserateunitid = 5 --unit: µg/kg/uur -> µg/kg/min
            THEN dose/60 
        END AS gamma
    FROM dosing
    WHERE
        -- medication given within 24 hours of ICU stay:
        start_time <= 24*60 AND stop_time >= 0
    ORDER BY admissionid, start_time
    """

    vaso = pd.read_gbq(sql_vaso, project_id=project_id, configuration=config_gbq)
    vaso.to_csv(ROOT + '\\saved_csvs\\vaso.csv')

# TODO get numericitems fixed (total and unvalidated)

def get_pf_ratio(project_id, config_gbq):
    # PF-ratio
    sql_pf = """
    WITH 
    
    numericitems_filtered AS(
    SELECT 
        admissionid,
        itemid,
        value,
        measuredat,
        unitid,
        registeredby
    FROM numericitems
    WHERE
        itemid IN (
            --Oxygen Flow settings without respiratory support
            8845, -- O2 l/min
            10387, --Zuurstof toediening (bloed)
            18587, --Zuurstof toediening

            --FiO2 settings on respiratory support
            6699, --FiO2 %: setting on Evita ventilator
            12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
            12369, --SET %O2: used with BiPap Vision ventilator
            16246, --Zephyros FiO2: Non-invasive ventilation

            7433, --PO2
            9996, --PO2 (bloed)
            21214 --PO2 (bloed) - kPa
        )
    ),

    numericitems_unvalidated_filtered AS(
    SELECT 
        admissionid,
        itemid,
        value,
        measuredat,
        unitid,
        registeredby
    FROM numericitems_unvalidated
    WHERE
        itemid IN (
            --Oxygen Flow settings without respiratory support
            8845, -- O2 l/min
            10387, --Zuurstof toediening (bloed)
            18587, --Zuurstof toediening

            --FiO2 settings on respiratory support
            6699, --FiO2 %: setting on Evita ventilator
            12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
            12369, --SET %O2: used with BiPap Vision ventilator
            16246, --Zephyros FiO2: Non-invasive ventilation

            7433, --PO2
            9996, --PO2 (bloed)
            21214 --PO2 (bloed) - kPa
        )
    ),

    numericitems_complete AS (
    SELECT * FROM numericitems_filtered
    UNION ALL
    SELECT * FROM numericitems_unvalidated_filtered
    ),
    
    fio2_table AS (
        SELECT n.admissionid,
            n.measuredat,
            l.valueid,
            l.value AS o2_device,
            CASE
                WHEN n.itemid IN (
                    --FiO2 settings on respiratory support
                    6699, --FiO2 %: setting on Evita ventilator
                    12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
                    12369, --SET %O2: used with BiPap Vision ventilator
                    16246 --Zephyros FiO2: Non-invasive ventilation
                ) THEN TRUE
                ELSE FALSE
            END AS ventilatory_support,
            n.itemid,
            CASE
                WHEN n.itemid IN (
                    --FiO2 settings on respiratory support
                    6699, --FiO2 %: setting on Evita ventilator
                    12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
                    12369, --SET %O2: used with BiPap Vision ventilator
                    16246 --Zephyros FiO2: Non-invasive ventilation
                ) THEN
                    CASE
                        WHEN NOT n.value IS NULL THEN n.value --use the settings
                        ELSE 0.21
                    END
                ELSE -- estimate the FiO2
                    CASE
                        WHEN l.valueid IN (
                            2, -- Nasaal
                            7 --O2-bril
                        ) THEN
                            CASE
                                WHEN n.value >= 1 AND n.value < 2 THEN 0.22
                                WHEN n.value >= 2 AND n.value < 3 THEN 0.25
                                WHEN n.value >= 3 AND n.value < 4 THEN 0.27
                                WHEN n.value >= 4 AND n.value < 5 THEN 0.30
                                WHEN n.value >= 5 THEN 0.35
                                ELSE 0.21
                            END
                        WHEN l.valueid IN (
                            1, --Diep Nasaal
                            3, --Kapje
                            8, --Kinnebak
                            9, --Nebulizer
                            4, --Kunstneus
                            18, --Spreekcanule
                            19 --Spreekklepje
                        ) THEN
                            CASE
                                WHEN n.value >= 1 AND n.value < 2 THEN 0.22 -- not defined by NICE
                                WHEN n.value >= 2 AND n.value < 3 THEN 0.25
                                WHEN n.value >= 3 AND n.value < 4 THEN 0.27
                                WHEN n.value >= 4 AND n.value < 5 THEN 0.30
                                WHEN n.value >= 5 AND n.value < 6 THEN 0.35
                                WHEN n.value >= 6 AND n.value < 7 THEN 0.40
                                WHEN n.value >= 7 AND n.value < 8 THEN 0.45
                                WHEN n.value >= 8 THEN 0.50
                                ELSE 0.21
                            END
                        WHEN l.valueid IN (
                            10, --Waterset
                            11, --Trach.stoma
                            13, --Ambu
                            14, --Guedel
                            15, --DL-tube
                            16, --CPAP
                            17 --Non-Rebreathing masker
                        ) THEN
                            CASE
                                WHEN n.value >= 6 AND n.value < 7 THEN 0.60
                                WHEN n.value >= 7 AND n.value < 8 THEN 0.70
                                WHEN n.value >= 8 AND n.value < 9 THEN 0.80
                                WHEN n.value >= 9 AND n.value < 10 THEN 0.85
                                WHEN n.value >= 10 THEN 0.90
                                ELSE 0.21
                            END
                        WHEN l.valueid IN (
                            12 --B.Lucht
                        ) THEN 0.21
                    ELSE 0.21
                END
            END AS fio2
        FROM numericitems_complete n
        LEFT JOIN admissions a ON
            n.admissionid = a.admissionid
        LEFT JOIN listitems l ON
            n.admissionid = l.admissionid AND
            n.measuredat = l.measuredat AND
            l.itemid = 8189 -- Toedieningsweg (Oxygen device)
        WHERE
            n.itemid IN (
                --Oxygen Flow settings without respiratory support
                8845, -- O2 l/min
                10387, --Zuurstof toediening (bloed)
                18587, --Zuurstof toediening

                --FiO2 settings on respiratory support
                6699, --FiO2 %: setting on Evita ventilator
                12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
                12369, --SET %O2: used with BiPap Vision ventilator
                16246 --Zephyros FiO2: Non-invasive ventilation
            )
        --measurements within 24 hours of ICU stay:
        AND (n.measuredat - a.admittedat) <= 1000*60*60*24 AND (n.measuredat - a.admittedat) >= 0
        AND n.value > 0 --ignore stand by values from Evita ventilator
    ),
    oxygenation AS (
        SELECT
            pao2.admissionid,
            CASE pao2.unitid
                WHEN 152 THEN ROUND(CAST(pao2.value * 7.50061683 AS NUMERIC), 1) -- Conversion: kPa to mmHg
                ELSE pao2.value
            END AS pao2,
            f.value AS specimen_source,
            CASE
                WHEN LOWER(pao2.registeredby) NOT LIKE '%systeem%' THEN TRUE
                ELSE FALSE
            END AS manual_entry,
            (pao2.measuredat - a.admittedat)/(1000*60) AS time,
            fio2_table.fio2,
            fio2_table.ventilatory_support,
            (fio2_table.measuredat - pao2.measuredat)/(60*1000) AS fio2_time_difference,
            ROW_NUMBER() OVER(
                PARTITION BY pao2.admissionid, pao2.measuredat
                ORDER BY
                    CASE
                        --FiO2 settings on respiratory support
                        WHEN fio2_table.itemid = 12279 THEN 1 --O2 concentratie --measurement by Servo-i/Servo-U ventilator
                        WHEN fio2_table.itemid = 6699 THEN 2 --FiO2 %: setting on Evita ventilator
                        WHEN fio2_table.itemid = 12369 THEN 3 --SET %O2: used with BiPap Vision ventilator
                        WHEN fio2_table.itemid = 16246 THEN 4--Zephyros FiO2: Non-invasive ventilation
                        --Oxygen Flow settings without respiratory support
                        WHEN fio2_table.itemid = 8845 THEN 5 -- O2 l/min
                        WHEN fio2_table.itemid = 10387 THEN 6 --Zuurstof toediening (bloed)
                        WHEN fio2_table.itemid = 18587 THEN 7 --Zuurstof toediening
                    END, --prefer ventilator measurements over manually entered O2 settings
                    CASE
                        WHEN fio2_table.measuredat <= pao2.measuredat THEN 1 --prefer FiO2 settings before blood gas measurement
                        ELSE 2
                    END,
                    ABS(fio2_table.measuredat - pao2.measuredat), --prefer FiO2 settings nearest to blood gas measurement
                    CASE
                        WHEN pao2.itemid = 21214 THEN 1 --PO2 (bloed) - kPa
                        WHEN pao2.itemid = 9996 THEN 2 --PO2 (bloed)
                        WHEN pao2.itemid = 7433 THEN 3 --PO2
                    END, --prefer PaO2 values from original measurement unit; same measurement could be reported twice
                    CASE
                        WHEN paco2.itemid = 21213 THEN 1--PCO2 (bloed) - kPa
                        WHEN paco2.itemid = 9990 THEN 2 --pCO2 (bloed)
                        WHEN  paco2.itemid = 6846 THEN 3 --PCO2
                    END, --prefer PaCO2 values from original measurement unit; same measurement could be reported twice
                    CASE
                        WHEN LOWER(f.value) LIKE '%art%' THEN 1
                        ELSE 2
                    END, --prefer samples that haven been specified as arterial
                    fio2_table.fio2 DESC --prefer highest (calculated) FiO2 for rare cases with multiple oxygen devices
                ) AS priority
            FROM numericitems_complete pao2
            LEFT JOIN admissions a ON
                pao2.admissionid = a.admissionid
            LEFT JOIN freetextitems f ON
                pao2.admissionid = f.admissionid AND
                pao2.measuredat = f.measuredat AND
                f.itemid = 11646 --Afname (bloed): source of specimen
            LEFT JOIN numericitems_complete paco2 ON
                pao2.admissionid = paco2.admissionid AND
                pao2.measuredat = paco2.measuredat AND
                paco2.itemid IN (
                    6846, --PCO2
                    9990, --pCO2 (bloed)
                    21213 --PCO2 (bloed) - kPa
                )
            LEFT JOIN fio2_table ON
                pao2.admissionid = fio2_table.admissionid AND
                fio2_table.measuredat > pao2.measuredat - 60*60*1000 AND --no earlier than 60 minutes before pao2 measurement
                fio2_table.measuredat < pao2.measuredat + 15*60*1000 --no later than 15 minutes after pao2 measurement
            WHERE
                pao2.itemid IN (
                    7433, --PO2
                    9996, --PO2 (bloed)
                    21214 --PO2 (bloed) - kPa
                    )
            --measurements within 24 hours of ICU stay (use 30 minutes before admission to allow for time differences):
            AND (pao2.measuredat - a.admittedat) <= 1000*60*60*24 AND (pao2.measuredat - a.admittedat) >= -(1000*60*30) AND
            (LOWER(f.value) LIKE '%art%' OR f.value IS NULL)  -- source is arterial or undefined (assume arterial)
    )
    SELECT * FROM oxygenation
    WHERE priority = 1
    """

    pf = pd.read_gbq(sql_pf, project_id=project_id, configuration=config_gbq)
    pf.to_csv(ROOT + '\\saved_csvs\\pf.csv')

def get_arterial_ph(project_id, config_gbq):
    # get arterial pH 
    sql_ph = """
    WITH 
    
    numericitems_filtered AS(
    SELECT
        n.admissionid,
        n.itemid,
        n.item,
        n.value,
        f.value AS specimen_source,
        n.registeredby,
        CASE
            WHEN LOWER(n.registeredby) NOT LIKE '%systeem%' THEN TRUE
            ELSE FALSE
        END AS manual_entry,
        (n.measuredat - a.admittedat)/(1000*60) AS time
    FROM numericitems n
    LEFT JOIN admissions a ON
        n.admissionid = a.admissionid
    LEFT JOIN freetextitems f ON
        n.admissionid = f.admissionid AND
        n.measuredat = f.measuredat AND
        f.itemid = 11646 --Afname (bloed): source of specimen (arterial)
    WHERE n.itemid IN (
        6848, --PH
        12310 --pH (bloed)
        )
    -- measurements within 24 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (n.measuredat - a.admittedat) <= 1000*60*60*24 AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    AND (LOWER(f.value) LIKE '%art%' OR f.value IS NULL) -- source is arterial or undefined (assume arterial)
    ),

    numericitems_unvalidated_filtered AS(
    SELECT
        n.admissionid,
        n.itemid,
        n.item,
        n.value,
        f.value AS specimen_source,
        n.registeredby,
        CASE
            WHEN LOWER(n.registeredby) NOT LIKE '%systeem%' THEN TRUE
            ELSE FALSE
        END AS manual_entry,
        (n.measuredat - a.admittedat)/(1000*60) AS time
    FROM numericitems_unvalidated n
    LEFT JOIN admissions a ON
        n.admissionid = a.admissionid
    LEFT JOIN freetextitems f ON
        n.admissionid = f.admissionid AND
        n.measuredat = f.measuredat AND
        f.itemid = 11646 --Afname (bloed): source of specimen (arterial)
    WHERE n.itemid IN (
        6848, --PH
        12310 --pH (bloed)
        )
    -- measurements within 24 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (n.measuredat - a.admittedat) <= 1000*60*60*24 AND (n.measuredat - a.admittedat) >= -(1000*60*30)
    AND (LOWER(f.value) LIKE '%art%' OR f.value IS NULL) -- source is arterial or undefined (assume arterial)
    ),

    numericitems_complete AS (
    SELECT * FROM numericitems_filtered
    UNION ALL
    SELECT * FROM numericitems_unvalidated_filtered
    )

    SELECT * FROM numericitems_complete
    """

    ph = pd.read_gbq(sql_ph, project_id=project_id, configuration=config_gbq)
    ph.to_csv(ROOT + '\\saved_csvs\\ph.csv')

def get_covariate_framework(project_id, config_gbq):
    sql_admissions = """
    SELECT admissionid FROM admissions
    """

    covariate_framework = pd.read_gbq(sql_admissions, project_id=project_id, configuration=config_gbq)
    covariate_framework.to_csv(ROOT + '\\saved_csvs\\covariate_framework.csv')

### BELOW MIMIC-IV ###

def get_sepsis_mimic(project_id, config_gbq): # TODO fix timing of sepsis3 classification
    sql_sepsis = """
        -- Creates a table with "onset" time of Sepsis-3 in the ICU.
    -- That is, the earliest time at which a patient had SOFA >= 2
    -- and suspicion of infection.
    -- As many variables used in SOFA are only collected in the ICU,
    -- this query can only define sepsis-3 onset within the ICU.

    -- extract rows with SOFA >= 2
    -- implicitly this assumes baseline SOFA was 0 before ICU admission.
    WITH sofa AS (
        SELECT stay_id
            , starttime, endtime
            , respiration_24hours AS respiration
            , coagulation_24hours AS coagulation
            , liver_24hours AS liver
            , cardiovascular_24hours AS cardiovascular
            , cns_24hours AS cns
            , renal_24hours AS renal
            , sofa_24hours AS sofa_score
        FROM `physionet-data.mimiciv_derived.sofa`
        WHERE sofa_24hours >= 2
    )

    , s1 AS (
        SELECT
            soi.subject_id
            , soi.stay_id
            -- suspicion columns
            , soi.ab_id
            , soi.antibiotic
            , soi.antibiotic_time
            , soi.culture_time
            , soi.suspected_infection
            , soi.suspected_infection_time
            , soi.specimen
            , soi.positive_culture
            -- sofa columns
            , starttime, endtime
            , respiration, coagulation, liver, cardiovascular, cns, renal
            , sofa_score
            -- All rows have an associated suspicion of infection event
            -- Therefore, Sepsis-3 is defined as SOFA >= 2.
            -- Implicitly, the baseline SOFA score is assumed to be zero,
            -- as we do not know if the patient has preexisting
            -- (acute or chronic) organ dysfunction before the onset
            -- of infection.
            , sofa_score >= 2 AND suspected_infection = 1 AS sepsis3
            -- subselect to the earliest suspicion/antibiotic/SOFA row
            , ROW_NUMBER() OVER
            (
                PARTITION BY soi.stay_id
                ORDER BY
                    suspected_infection_time, antibiotic_time, culture_time, endtime
            ) AS rn_sus
        FROM `physionet-data.mimiciv_derived.suspicion_of_infection` AS soi
        INNER JOIN sofa
            ON soi.stay_id = sofa.stay_id
                AND sofa.endtime >= DATETIME_SUB(
                    soi.suspected_infection_time, INTERVAL '48' HOUR
                )
                AND sofa.endtime <= DATETIME_ADD(
                    soi.suspected_infection_time, INTERVAL '24' HOUR
                )
        -- only include in-ICU rows
        WHERE soi.stay_id IS NOT NULL
    )

    SELECT
        s1.subject_id 
        , s1.stay_id
        -- note: there may be more than one antibiotic given at this time
        , s1.antibiotic_time
        -- culture times may be dates, rather than times
        , s1.culture_time
        , s1.suspected_infection_time
        -- endtime is latest time at which the SOFA score is valid
        , s1.endtime AS sofa_time
        , s1.sofa_score
        , s1.respiration, coagulation, liver, cardiovascular, cns, renal
        , s1.sepsis3
        , icu.icu_intime -- added
    FROM s1
    LEFT JOIN `physionet-data.mimiciv_derived.icustay_detail` AS icu ON s1.stay_id = icu.stay_id -- added
    WHERE 
        rn_sus = 1
    AND -- added
        TIMESTAMP_DIFF(s1.suspected_infection_time, icu.icu_intime, HOUR) < 24
    """
        
    sepsis = pd.read_gbq(sql_sepsis, project_id=project_id, configuration=config_gbq)
    sepsis.to_csv(ROOT + '\\saved_csvs\\mimic_sepsis.csv')

def get_mortality_mimic(project_id, config_gbq):
    sql_mortality = """
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        gender,
        dod,
        admission_age,
        icu_intime,
        icu_outtime,
    FROM `physionet-data.mimiciv_derived.icustay_detail`
    """

    mortality = pd.read_gbq(sql_mortality, project_id=project_id, configuration=config_gbq)
    mortality.to_csv(ROOT + '\\saved_csvs\\mimic_mortality.csv')

def get_crp_mimic(project_id, config_gbq):
    sql_crp = """
    SELECT
        subject_id,
        hadm_id,
        charttime,
        crp
    FROM `physionet-data.mimiciv_derived.inflammation`
    """

    crp = pd.read_gbq(sql_crp, project_id=project_id, configuration=config_gbq)
    crp.to_csv(ROOT + '\\saved_csvs\\mimic_crp.csv')

def get_steroids_mimic(project_id, config_gbq): # TODO change to within 24h
    sql_steroids = """
    WITH aux_steroid AS (
        SELECT
            emar.subject_id
             , emar.hadm_id
             , emar.emar_id
             , medication
             , dose_given
             , emar.charttime
             , icu.stay_id
             , icu.icu_intime
             , icu.icu_outtime
             , icu.los_icu -- icu stay in days
             ,CASE
                  WHEN medication = 'Dexamethasone' THEN CAST(dose_given AS DECIMAL) * 5
                  WHEN medication = 'Hydrocortisone' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'Hydrocortisone Na Succ.' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'INV-Dexamethasone' THEN CAST(dose_given AS DECIMAL) * 5
                  WHEN medication = 'MethylPREDNISolone Sodium Succ' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'Methylprednisolone' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'Methylprednisolone ACETATE' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'Methylprednisolone Na Succ Desensitization' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'PredniSONE' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'Prednisone' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'dexamethasone' THEN CAST(dose_given AS DECIMAL) * 5
                  WHEN medication = 'hydrocorTISone' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'hydrocorTISone Sod Succ (PF)' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'hydrocorTISone Sod Succinate' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'hydrocortisone-acetic acid' THEN CAST(dose_given AS DECIMAL) * 0.2
                  WHEN medication = 'methylPREDNISolone acetate' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'predniSONE' THEN CAST(dose_given AS DECIMAL) * 1
                  WHEN medication = 'predniSONE Intensol' THEN CAST(dose_given AS DECIMAL) * 1
            END AS methylprednisolone_equivalent


        FROM `physionet-data.mimiciv_hosp.emar` emar

                 LEFT JOIN `physionet-data.mimiciv_hosp.emar_detail` AS detail
                           ON emar.emar_id = detail.emar_id

                 LEFT JOIN `physionet-data.mimiciv_derived.icustay_detail` icu
                           ON emar.hadm_id = icu.hadm_id


        WHERE medication IN (
                             'Dexamethasone',
                             'Hydrocortisone',
                             'Hydrocortisone Na Succ.',
                             'INV-Dexamethasone',
                             'MethylPREDNISolone Sodium Succ',
                             'Methylprednisolone',
                             'Methylprednisolone ACETATE',
                             'Methylprednisolone Na Succ Desensitization',
                             'PredniSONE',
                             'Prednisone',
                             'dexamethasone',
                             'hydrocorTISone',
                             'hydrocorTISone Sod Succ (PF)',
                             'hydrocorTISone Sod Succinate',
                             'hydrocortisone-acetic acid',
                             'methylPREDNISolone acetate',
                             'predniSONE',
                             'predniSONE Intensol'
            )
          AND event_txt IN (
                            'Delayed Administered',
                            'Administered in Other Location',
                            'Documented in O.R. Holding',
                            'Partial Administered',
                            'Delayed Started',
                            'Confirmed',
                            'Administered',
                            'Started',
                            'Restarted'
            )
          AND detail.dose_given <> "_"
          AND detail.dose_given <> "___"
          AND TIMESTAMP_DIFF(icu_outtime, charttime, HOUR) > 0
          AND TIMESTAMP_DIFF(charttime, icu_intime, HOUR) > 0
          AND TIMESTAMP_DIFF(charttime, icu_intime, HOUR) < 72 -- only keep steroids given within 72h of icu admission

    )

    SELECT
        --hadm_id,
        stay_id
         , SUM(methylprednisolone_equivalent) AS methylprednisolone_equivalent_total
         , SUM(methylprednisolone_equivalent)/MAX(los_icu) AS methylprednisolone_equivalent_normalized_by_icu_los
    FROM aux_steroid
    GROUP BY stay_id
    """

    steroids = pd.read_gbq(sql_steroids, project_id=project_id, configuration=config_gbq)
    steroids.to_csv(ROOT + '\\saved_csvs\\mimic_steroids.csv')

def get_max_gamma_mimic(project_id, config_gbq):
    sql_max_gamma = """
    WITH FirstRows_ned AS (
    SELECT
        ned.stay_id,
        ned.starttime,
        ned.endtime,
        ned.norepinephrine_equivalent_dose,
        FIRST_VALUE(ned.starttime) OVER (PARTITION BY ned.stay_id ORDER BY ned.starttime) AS first_starttime,
    FROM `physionet-data.mimiciv_derived.first_day_bg` fdbg
    FULL OUTER JOIN `physionet-data.mimiciv_derived.norepinephrine_equivalent_dose` ned ON fdbg.stay_id = ned.stay_id
    WHERE fdbg.stay_id IS NOT NULL
    ORDER BY fdbg.stay_id, ned.starttime
    )
    SELECT
    stay_id,
    MAX(norepinephrine_equivalent_dose) AS max_gamma_vasopressor,
    FROM FirstRows_ned
    WHERE TIMESTAMP_DIFF(endtime, first_starttime, HOUR) <= 24
    GROUP BY stay_id
    ORDER BY stay_id
    """

    max_gamma = pd.read_gbq(sql_max_gamma, project_id=project_id, configuration=config_gbq)
    max_gamma.to_csv(ROOT + '\\saved_csvs\\mimic_max_gamma.csv')

# def get_blood_gas_mimic(project_id, config_gbq):
#     sql_blood_gas = """
#     -- lactate_max, sodium_max, potassium_max, temperature_max, bicarbonate_max, glucose_max, ph_min
#     SELECT
#     fdbg.stay_id,
#     MAX(fdbg.lactate_max) AS max_lactate,
#     MAX(fdbg.sodium_max) AS max_sodium,
#     MAX(fdbg.potassium_max) AS max_potassium,
#     MAX(fdbg.temperature_max) AS max_temperature,
#     MIN(fdbg.bicarbonate_max) AS min_bicarbonate,
#     MAX(fdbg.glucose_max) AS max_glucose,
#     MIN(fdbg.ph_min) AS min_ph,
#     FROM `physionet-data.mimiciv_derived.first_day_bg` fdbg
#     LEFT JOIN `physionet-data.mimiciv_derived.bg` bg ON bg.subject_id = fdbg.subject_id
#     INNER JOIN `physionet-data.mimiciv_hosp.patients` p ON fdbg.subject_id = p.subject_id
#     INNER JOIN `physionet-data.mimiciv_derived.inflammation` i ON fdbg.subject_id = i.subject_id
#     GROUP BY fdbg.stay_id
#     -- HAVING max_lactate IS NOT NULL
#     --   AND max_sodium IS NOT NULL
#     --   AND max_potassium IS NOT NULL
#     --   AND max_temperature IS NOT NULL
#     --   AND max_bicarbonate IS NOT NULL
#     --   AND max_glucose IS NOT NULL
#     --   AND min_ph IS NOT NULL
#     ORDER BY fdbg.stay_id
#     """

#     blood_gas = pd.read_gbq(sql_blood_gas, project_id=project_id, configuration=config_gbq)
#     blood_gas.to_csv(ROOT + '\\saved_csvs\\mimic_blood_gas.csv')

def get_blood_gas_mimic(project_id, config_gbq):
    sql_blood_gas = """
    SELECT
        i.stay_id,
        MIN(pao2fio2ratio) AS min_pf_ratio,
        AVG(i.hadm_id),
        MAX(lactate) AS max_lactate,
        --MAX(sodium) AS max_sodium,
        --MAX(potassium) AS max_potassium,
        --MAX(temperature) AS max_temperature,
        --MIN(bicarbonate) AS min_bicarbonate,
        --MAX(glucose) AS max_glucose,
        MIN(ph) AS min_ph,
        MAX(TIMESTAMP_DIFF(i.intime, bg.charttime, HOUR)) as time_diff,
        count(*) as rows_per_hadm
    FROM `physionet-data.mimiciv_derived.bg` bg
    INNER JOIN `physionet-data.mimiciv_icu.icustays` i ON bg.subject_id = i.subject_id
    WHERE TIMESTAMP_DIFF(i.intime, bg.charttime, HOUR) BETWEEN -8 AND 24
    GROUP BY i.stay_id
    ORDER BY i.stay_id
    """

    blood_gas = pd.read_gbq(sql_blood_gas, project_id=project_id, configuration=config_gbq)
    blood_gas.to_csv(ROOT + '\\saved_csvs\\mimic_blood_gas_2.csv')



def get_features_mimic(project_id, config_gbq):
    sql_features_mimic = """
    -- heart_rate_mean, bun_max, creatinine_max, abs_lymphocytes (min and max), leucocytes (wbc) (min and max). pf-ratio moved to blood gas mimic
    SELECT
    fdlab.stay_id,
    fdlab.sodium_max AS max_sodium,
    fdlab.potassium_max AS max_potassium,
    fdlab.glucose_max AS max_glucose,
    fdlab.bicarbonate_min as min_bicarbonate,
    fdlab.bun_max AS max_bun,
    fdlab.creatinine_max AS max_creatinine,
    fdlab.abs_lymphocytes_max AS max_lymphocytes,
    fdlab.abs_lymphocytes_min AS min_lymphocytes,
    fdlab.wbc_max AS max_wbc,
    fdv.heart_rate_mean AS mean_heart_rate, -- TODO change this to median
    fdv.temperature_max AS max_temperature
    FROM `physionet-data.mimiciv_derived.first_day_lab` AS fdlab
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_vitalsign` AS fdv ON fdlab.stay_id = fdv.stay_id
    --GROUP BY fdbg.stay_id
    -- HAVING max_bun IS NOT NULL
    --   AND max_lactate IS NOT NULL
    --   AND max_lymphocytes IS NOT NULL
    --   AND min_lymphocytes IS NOT NULL
    --   AND min_wbc IS NOT NULL
    --   AND max_wbc IS NOT NULL
    --   AND max_pf_ratio IS NOT NULL
    --   AND mean_heart_rate IS NOT NULL
    ORDER BY fdlab.stay_id
    """

    features = pd.read_gbq(sql_features_mimic, project_id=project_id, configuration=config_gbq)
    features.to_csv(ROOT + '\\saved_csvs\\mimic_features.csv')


def get_demographics_mimic(project_id, config_gbq):
    sql_demographics_mimic = """
    -- weight, height, anchor_age, gender
    WITH RankedData AS (
    SELECT
        wd.stay_id,
        wd.weight,
        h.height,
        i.gender,
        i.admission_age as age,
        --ROW_NUMBER() OVER (PARTITION BY h.stay_id ORDER BY wd.starttime) AS rn
    FROM `physionet-data.mimiciv_derived.first_day_bg` fdbg
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_weight` wd ON fdbg.stay_id = wd.stay_id --changed from INNER to test
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_height` h ON fdbg.stay_id = h.stay_id
    LEFT JOIN `physionet-data.mimiciv_derived.icustay_detail` i ON fdbg.stay_id = i.stay_id
    --WHERE wd.weight_type = "admit"
    )
    SELECT
        stay_id,
        weight,
        height,
        gender,
        age
    FROM RankedData
    --WHERE rn = 1
    ORDER BY stay_id
    """

    demographics = pd.read_gbq(sql_demographics_mimic, project_id=project_id, configuration=config_gbq)
    demographics.to_csv(ROOT + '\\saved_csvs\\mimic_demographics.csv')

def get_ventilation_mimic(project_id, config_gbq):
    sql_ventilation_mimic = """
    -- ventilation
    SELECT fdbg.stay_id,
    CAST(
        LOGICAL_AND(
        CASE 
            WHEN v.ventilation_status IN ("InvasiveVent", "NonInvasiveVent") THEN TRUE
            ELSE FALSE
        END
        ) AS INT64
    ) AS ventilation
    FROM `physionet-data.mimiciv_derived.ventilation` v
    FULL OUTER JOIN `physionet-data.mimiciv_derived.first_day_bg` fdbg ON v.stay_id = fdbg.stay_id
    GROUP BY fdbg.stay_id
    ORDER BY fdbg.stay_id
    """
    
    ventilation = pd.read_gbq(sql_ventilation_mimic, project_id=project_id, configuration=config_gbq)
    ventilation.to_csv(ROOT + '\\saved_csvs\\mimic_ventilation.csv')