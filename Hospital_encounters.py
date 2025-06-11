import pandas as pd
from collections import Counter
from scipy import stats

def analyze_daily_deaths(file_path="Øya_2_hospitalencounters.csv"):
    df = pd.read_csv(file_path, na_values=[""])

    # Filter out test patient
    df = df[df["PatientPseudoKey"] != 2384]

    # Filter out blank DeathDate values before parsing
    df = df[df["DeathDate"].notna()]
    df["DeathDate"] = pd.to_datetime(df["DeathDate"], dayfirst=True, errors="coerce")
    df = df[df["DeathDate"].notna()]

    # Drop duplicate PatientPseudoKey to count each patient once
    df = df.drop_duplicates(subset=["PatientPseudoKey"])
    print(f"✅ Parsed valid death dates (unique patients): {len(df)}")

    # Get only date part
    df["DeathDateOnly"] = df["DeathDate"].dt.date

    # Count deaths per day (unique patients)
    daily_deaths = df["DeathDateOnly"].value_counts().sort_index()

    if daily_deaths.empty:
        print("\n💀 Deaths per day:\nNo valid death dates found.")
        return daily_deaths

    min_val = daily_deaths.min()
    max_val = daily_deaths.max()
    avg_val = daily_deaths.mean()
    mode_result = stats.mode(daily_deaths, keepdims=True)
    mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else float("nan")

    print(f"\n💀 Deaths per day (unique patients):")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")
    print(f"Average: {avg_val:.2f}")
    print(f"Mode: {mode_val}")

    return daily_deaths.sort_index()


def analyze_daily_admissions(file_path="Øya_2_hospitalencounters.csv"):
    df = pd.read_csv(file_path)

    # Filter out test patient and keep only relevant source
    df = df[df["PatientPseudoKey"] != 2384]
    df = df[df["AdmissionSource"] == "Bosted/arbeidsted"]

    # Parse EncounterStart to date
    df["EncounterStart"] = pd.to_datetime(df["EncounterStart"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["AdmissionDate"] = df["EncounterStart"].dt.date

    df = df.drop_duplicates(subset=["PatientPseudoKey", "AdmissionDate"])

    # Count admissions per day
    daily_counts = df["AdmissionDate"].value_counts().sort_index()

    min_val = daily_counts.min()
    max_val = daily_counts.max()
    avg_val = daily_counts.mean()
    mode_val = stats.mode(daily_counts, keepdims=True).mode[0]

    print(f"📆 Admissions from home per day:")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")
    print(f"Average: {avg_val:.2f}")
    print(f"Mode: {mode_val}")
    
    return daily_counts.sort_index()

def count_unique_patients(file_path="Øya_2_hospitalencounters.csv"):
    df = pd.read_csv(file_path, na_values=[""])

    # Filter out test patient
    df = df[df["PatientPseudoKey"] != 2384]

    total_unique = df["PatientPseudoKey"].nunique()

    # Parse and filter DeathDate properly
    if "DeathDate" in df.columns:
        df["DeathDate"] = pd.to_datetime(df["DeathDate"], dayfirst=True, errors="coerce")
        df = df[df["DeathDate"].notna()]
        df = df.drop_duplicates(subset=["PatientPseudoKey"])
        deceased_unique = df["PatientPseudoKey"].nunique()
    else:
        print("⚠️ 'DeathDate' column not found in the dataset.")
        deceased_unique = None

    print(f"🧍 Total unique patients: {total_unique}")
    print(f"💀 Unique deceased patients: {deceased_unique}")

    return total_unique, deceased_unique





# New function: analyze_daily_admissions_byCFS
def analyze_daily_admissions_byCFS(
    hospital_path="Øya_2_hospitalencounters.csv",
    oya_path="Øya_encounters.csv",
    adl_path="Øya_2_ADL.csv",
    measurement_name="R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT"
):
    import numpy as np

    # Load and preprocess hospital data
    hosp_df = pd.read_csv(hospital_path)
    hosp_df = hosp_df[hosp_df["PatientPseudoKey"] != 2384]

    # Filter by AdmissionSource
    hosp_df = hosp_df[hosp_df["AdmissionSource"] == "Bosted/arbeidsted"]

    # Load vedtak data
    vedtak_df = pd.read_csv("Øya_decisions.csv")
    vedtak_df = vedtak_df[vedtak_df["DecisionTemplate"] == "Vedtak om langtidsopphold i institusjon"]
    vedtak_df["DecisionValidDate"] = pd.to_datetime(vedtak_df["DecisionValidDate"], format="%Y-%m-%d", errors="coerce")

    # Parse EncounterStart/End after filtering
    hosp_df["EncounterStart"] = pd.to_datetime(hosp_df["EncounterStart"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    hosp_df["EncounterEnd"] = pd.to_datetime(hosp_df["EncounterEnd"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    hosp_df["EncounterStartDate"] = hosp_df["EncounterStart"].dt.date
    hosp_df["EncounterEndDate"] = hosp_df["EncounterEnd"].dt.date
    hosp_df = hosp_df.drop_duplicates(subset=["PatientPseudoKey", "EncounterStart", "EncounterEnd"])

    # Mark patients to exclude based on vedtak
    def has_vedtak_before_encounter(row):
        patient_decisions = vedtak_df[vedtak_df["PatientPseudoKey"] == row["PatientPseudoKey"]]
        return any(patient_decisions["DecisionValidDate"] <= row["EncounterEnd"])

    hosp_df = hosp_df[~hosp_df.apply(has_vedtak_before_encounter, axis=1)]

    # Load and preprocess Øya encounters
    oya_df = pd.read_csv(oya_path)
    oya_df = oya_df[oya_df["PatientPseudoKey"] != 2384]
    oya_df["EncounterEnd"] = pd.to_datetime(oya_df["EncounterEnd"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    oya_df["EncounterEndDate"] = oya_df["EncounterEnd"].dt.date

    # Load and preprocess ADL data
    adl_df = pd.read_csv(adl_path)
    adl_df = adl_df[adl_df["PatientPseudoKey"] != 2384]
    adl_df = adl_df[adl_df["MeasurementName"] == measurement_name]
    adl_df["MeasurementTime"] = pd.to_datetime(adl_df["MeasurementTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    adl_df = adl_df.dropna(subset=["MeasurementTime", "Value"])
    adl_df["Value"] = pd.to_numeric(adl_df["Value"], errors="coerce")
    adl_df = adl_df.dropna(subset=["Value"])

    # Bin ADL values into 9 equal-width deciles
    adl_df["Decile"] = pd.cut(adl_df["Value"], 9, labels=False)
    adl_df["DecileLabel"] = pd.cut(adl_df["Value"], 9).astype(str)

    # Assign decile to each hospital encounter based on closest ADL to EncounterEnd
    hosp_df["ADL_DecileLabel"] = None
    for i, row in hosp_df.iterrows():
        patient_adls = adl_df[adl_df["PatientPseudoKey"] == row["PatientPseudoKey"]]
        if not patient_adls.empty and pd.notna(row["EncounterEnd"]):
            patient_adls = patient_adls.copy()
            patient_adls["TimeDiff"] = (patient_adls["MeasurementTime"] - row["EncounterEnd"]).abs()
            closest = patient_adls.sort_values("TimeDiff").iloc[0]
            hosp_df.at[i, "ADL_DecileLabel"] = closest["DecileLabel"]

    hosp_df = hosp_df.dropna(subset=["ADL_DecileLabel"])

    # Determine readmissions
    readmission_flags = []
    for idx, row in hosp_df.iterrows():
        patient_id = row["PatientPseudoKey"]
        current_start = row["EncounterStartDate"]

        other_hosp_ends = hosp_df[
            (hosp_df["PatientPseudoKey"] == patient_id) &
            (hosp_df["EncounterEndDate"] < current_start)
        ]["EncounterEndDate"]

        other_oya_ends = oya_df[
            (oya_df["PatientPseudoKey"] == patient_id) &
            (oya_df["EncounterEndDate"] < current_start)
        ]["EncounterEndDate"]

        readmitted = False
        for past_date in pd.concat([other_hosp_ends, other_oya_ends]):
            if (current_start - past_date).days <= 30:
                readmitted = True
                break

        readmission_flags.append("Readmission" if readmitted else "New admission")

    hosp_df["AdmissionType"] = readmission_flags

    # Find min/max date for EncounterStartDate
    min_date = hosp_df["EncounterStartDate"].min()
    max_date = hosp_df["EncounterStartDate"].max()
    date_range = pd.date_range(start=min_date, end=max_date).date

    # Group by decile and admission type
    results = {}

    for decile in hosp_df["ADL_DecileLabel"].unique():
        results[decile] = {}
        for adm_type in ["New admission", "Readmission"]:
            subset = hosp_df[
                (hosp_df["ADL_DecileLabel"] == decile) &
                (hosp_df["AdmissionType"] == adm_type)
            ]
            counts = subset["EncounterStartDate"].value_counts()
            counts = counts.reindex(date_range, fill_value=0).sort_index()
            total_admissions = len(subset)
            if not counts.empty:
                min_val = counts.min()
                max_val = counts.max()
                avg_val = counts.mean()
                mode_result = stats.mode(counts, keepdims=True)
                mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else float("nan")
            else:
                min_val = max_val = avg_val = mode_val = 0
            results[decile][adm_type] = {
                "Minimum": min_val,
                "Maximum": max_val,
                "Average": round(avg_val, 2),
                "Mode": mode_val,
                "TotalAdmissions": total_admissions
            }

    print("\n📊 Daily hospital admissions by ADL decile and admission type:")
    for decile in sorted(results.keys()):
        print(f"\nDecile: {decile}")
        for adm_type in ["New admission", "Readmission"]:
            stats_ = results[decile][adm_type]
            print(f"  {adm_type}: Min={stats_['Minimum']}, Max={stats_['Maximum']}, Avg={stats_['Average']}, Mode={stats_['Mode']}, Total={stats_['TotalAdmissions']}")

def analyze_initial_adl_distribution(
    adl_path="Øya_2_ADL.csv",
    measurement_name="R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT"
):
    adl_df = pd.read_csv(adl_path)

    # Filter out test patient and invalid entries
    adl_df = adl_df[adl_df["PatientPseudoKey"] != 2384]
    adl_df = adl_df[adl_df["MeasurementName"] == measurement_name]
    adl_df["MeasurementTime"] = pd.to_datetime(adl_df["MeasurementTime"], errors="coerce")
    adl_df = adl_df.dropna(subset=["MeasurementTime", "Value"])
    adl_df["Value"] = pd.to_numeric(adl_df["Value"], errors="coerce")
    adl_df = adl_df.dropna(subset=["Value"])

    # Find earliest measurement for each patient
    earliest = adl_df.sort_values("MeasurementTime").groupby("PatientPseudoKey").first().reset_index()

    # Create 9 equal-width bins
    earliest["DecileLabel"] = pd.cut(earliest["Value"], 9).astype(str)
    counts = earliest["DecileLabel"].value_counts().sort_index()

    print("📊 Initial ADL decile distribution (based on earliest measurement):")
    for label, count in counts.items():
        print(f"  {label}: {count} patients")


if __name__ == "__main__":
    analyze_daily_admissions()
    count_unique_patients()
    analyze_daily_deaths()
    analyze_daily_admissions_byCFS()
    analyze_initial_adl_distribution()