import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from Oya_encounters import load_and_filter_encounters


def analyze_cfs_outcomes(cfs_path="\u00d8ya_CFS.csv", encounters_path="\u00d8ya_encounters.csv"):
    # Load CFS data
    cfs_df = pd.read_csv(cfs_path)
    cfs_df = cfs_df[cfs_df["PatientPseudoKey"] != 2384]  # Remove test patient
    cfs_df["TakenInstant"] = pd.to_datetime(cfs_df["TakenInstant"], errors="coerce")

    # Load and filter encounters
    encounters_df = load_and_filter_encounters(encounters_path)
    encounters_df["DischargeInstant"] = pd.to_datetime(encounters_df["DischargeInstant"], errors="coerce")

    # Get all distinct DischargeDispositions
    all_dispositions = encounters_df["DischargeDisposition"].dropna().unique().tolist()

    # Prepare result structure: dict of dicts
    result_dict = defaultdict(lambda: {"Antall": 0, **{disp: 0 for disp in all_dispositions}})

    # Iterate through each CFS entry
    for _, row in cfs_df.iterrows():
        patient_id = row["PatientPseudoKey"]
        cfs = row["CFS"]
        if pd.isna(row["TakenInstant"]):
            continue

        taken_date = row["TakenInstant"].normalize()  # round to midnight for consistent comparison

        # Find matching encounters
        patient_encounters = encounters_df[encounters_df["PatientPseudoKey"] == patient_id]
        future_encounters = patient_encounters[patient_encounters["DischargeInstant"] >= taken_date]

        if not future_encounters.empty:
            first_encounter = future_encounters.sort_values(by="DischargeInstant").iloc[0]
            disposition = first_encounter["DischargeDisposition"]

            # Update result_dict
            result_dict[cfs]["Antall"] += 1
            if disposition in result_dict[cfs]:
                result_dict[cfs][disposition] += 1
            else:
                result_dict[cfs][disposition] = 1  # handle unexpected disposition

    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(result_dict, orient="index").fillna(0).reset_index()
    result_df.rename(columns={"index": "CFS"}, inplace=True)
    result_df = result_df.sort_values(by="CFS")

    # Add a final row summing all columns (excluding CFS)
    sum_row = result_df.drop(columns=["CFS"]).sum()
    sum_row["CFS"] = "sum"
    result_df = pd.concat([result_df, pd.DataFrame([sum_row])], ignore_index=True)

    return result_df


def analyze_cfs_destinations_by_intervals(cfs_path="Øya_CFS.csv", encounters_path="Øya_encounters.csv"):
    # Load and preprocess CFS data
    cfs_df = pd.read_csv(cfs_path)
    cfs_df = cfs_df[cfs_df["PatientPseudoKey"] != 2384]
    cfs_df["TakenInstant"] = pd.to_datetime(cfs_df["TakenInstant"], errors="coerce")

    # Load and preprocess encounter data
    encounters_df = pd.read_csv(encounters_path)
    encounters_df = encounters_df[encounters_df["PatientPseudoKey"] != 2384]
    encounters_df = encounters_df[encounters_df["EncounterType"] == "Sykehuskontakt"]
    encounters_df["DischargeInstant"] = pd.to_datetime(encounters_df["DischargeInstant"], errors="coerce")

    # Group mappings
    group_hospital = {
        "Annen helseinstitusjon innen spesialisthelsetjenesten",
        "Annen helseinstitusjon innenfor spesialisthelsetjenesten",  # Added this variant
        "Somatisk sykehus STO",
        "Psykiatrisk sykehus STO"
    }
    group_other = {
        "*Unspecified",
        "Annet",
        "Annen institusjon (ikke helse)"
    }

    def group_destination(dest):
        if pd.isna(dest):
            return None
        dest = dest.strip()
        if dest in group_hospital:
            return "Hospital"
        elif dest in group_other:
            return "Other/Unspecified"
        else:
            return dest

    encounters_df["DischargeDestGroup"] = encounters_df["DischargeDestination"].apply(group_destination)

    # Prepare result structure
    unique_destinations = encounters_df["DischargeDestGroup"].dropna().unique().tolist()
    result_dict = defaultdict(lambda: {"Antall": 0, **{d: 0 for d in unique_destinations}})

    date_diffs = []

    # Process each patient
    for patient_id, patient_cfs in cfs_df.groupby("PatientPseudoKey"):
        patient_cfs = patient_cfs.dropna(subset=["TakenInstant"]).sort_values("TakenInstant")
        patient_encounters = encounters_df[encounters_df["PatientPseudoKey"] == patient_id]

        cfs_times = patient_cfs["TakenInstant"].tolist()
        cfs_scores = patient_cfs["CFS"].tolist()

        for i, (cfs, start_time) in enumerate(zip(cfs_scores, cfs_times)):
            start_time = start_time.normalize()
            end_time = cfs_times[i + 1].normalize() if i + 1 < len(cfs_times) else None

            if end_time:
                relevant_encounters = patient_encounters[
                    (patient_encounters["DischargeInstant"] >= start_time) &
                    (patient_encounters["DischargeInstant"] < end_time)
                ]
            else:
                relevant_encounters = patient_encounters[
                    (patient_encounters["DischargeInstant"] >= start_time)
                ]

            for _, enc in relevant_encounters.iterrows():
                dest_group = enc["DischargeDestGroup"]
                date_diff = (enc["DischargeInstant"].normalize() - start_time).days
                date_diffs.append(date_diff)
                if pd.notna(dest_group):
                    result_dict[cfs]["Antall"] += 1
                    result_dict[cfs][dest_group] += 1

    # Print average date difference if available
    if date_diffs:
        avg_days_diff = sum(date_diffs) / len(date_diffs)
        print(f"📏 Average days between CFS measurement and discharge: {avg_days_diff:.2f} days")
    else:
        print("⚠️ No valid CFS-discharge matches to calculate average days difference.")

    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(result_dict, orient="index").fillna(0).reset_index()
    result_df.rename(columns={"index": "CFS"}, inplace=True)
    result_df = result_df.sort_values(by="CFS")

    # Add sum row
    sum_row = result_df.drop(columns=["CFS"]).sum()
    sum_row["CFS"] = "sum"
    result_df = pd.concat([result_df, pd.DataFrame([sum_row])], ignore_index=True)

    return result_df


# Updated function for plotting CFS vs ADL
def plot_cfs_vs_adl(
    cfs_path="Øya_CFS.csv",
    adl_path="Øya_2_ADL.csv",
    encounters_path="Øya_encounters.csv",
    measurement_name="ADL",  #"ADL", "Total score ADL", "R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT"
    onlyUseLast50Days=True,
):
    # Load and preprocess data
    cfs_df = pd.read_csv(cfs_path)
    adl_df = pd.read_csv(adl_path)
    encounters_df = load_and_filter_encounters(encounters_path)

    # Filter and convert date fields
    cfs_df = cfs_df[cfs_df["PatientPseudoKey"] != 2384]
    adl_df = adl_df[adl_df["PatientPseudoKey"] != 2384]
    encounters_df = encounters_df[encounters_df["PatientPseudoKey"] != 2384]
    encounters_df["DischargeInstant"] = pd.to_datetime(encounters_df["DischargeInstant"], errors="coerce")
    cfs_df["TakenInstant"] = pd.to_datetime(cfs_df["TakenInstant"], errors="coerce")
    adl_df["MeasurementTime"] = pd.to_datetime(adl_df["MeasurementTime"], errors="coerce")
    adl_df = adl_df[adl_df["MeasurementName"] == measurement_name]
    adl_df = adl_df.dropna(subset=["MeasurementTime", "Value"])
    adl_df["Value"] = pd.to_numeric(adl_df["Value"], errors="coerce")

    results = []

    for _, encounter in encounters_df.iterrows():
        patient_id = encounter["PatientPseudoKey"]
        discharge_time = encounter["DischargeInstant"]

        if pd.isna(discharge_time):
            continue

        # Latest CFS before or at discharge
        patient_cfs = cfs_df[(cfs_df["PatientPseudoKey"] == patient_id) & 
                             (cfs_df["TakenInstant"] <= discharge_time)]
        if patient_cfs.empty:
            continue
        latest_cfs_row = patient_cfs.sort_values("TakenInstant").iloc[-1]
        latest_cfs = latest_cfs_row["CFS"]
        cfs_diff_days = (discharge_time - latest_cfs_row["TakenInstant"]).days

        # Latest ADL before or at discharge
        patient_adl = adl_df[(adl_df["PatientPseudoKey"] == patient_id) & 
                             (adl_df["MeasurementTime"] <= discharge_time)]
        if patient_adl.empty:
            continue
        latest_adl_row = patient_adl.sort_values("MeasurementTime").iloc[-1]
        latest_adl = latest_adl_row["Value"]
        adl_diff_days = (discharge_time - latest_adl_row["MeasurementTime"]).days

        if onlyUseLast50Days and (cfs_diff_days > 50 or adl_diff_days > 50):
            continue

        results.append({"CFS": latest_cfs, "ADL": latest_adl})

    if not results:
        print("⚠️ No valid CFS and ADL matches found.")
        return

    result_df = pd.DataFrame(results)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(result_df["CFS"], result_df["ADL"], alpha=0.7)
    plt.xlabel("CFS Score")
    plt.ylabel(f"{measurement_name} Value")
    plt.title(f"CFS vs {measurement_name} (based on encounter discharge)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def analyze_adl_outcomes_by_decile(adl_path="Øya_2_ADL.csv", encounters_path="Øya_encounters.csv", measurement_name="R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT", accept_scores_after=True):
    #"ADL", "Total score ADL", "R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT"
    # Load and preprocess ADL data
    adl_df = pd.read_csv(adl_path)
    adl_df = adl_df[adl_df["PatientPseudoKey"] != 2384]
    adl_df = adl_df[adl_df["MeasurementName"] == measurement_name]
    adl_df["MeasurementTime"] = pd.to_datetime(adl_df["MeasurementTime"], errors="coerce")
    adl_df = adl_df.dropna(subset=["MeasurementTime", "Value"])
    adl_df["Value"] = pd.to_numeric(adl_df["Value"], errors="coerce")
    adl_df = adl_df.dropna(subset=["Value"])

    # Load and preprocess encounters
    encounters_df = pd.read_csv(encounters_path)
    encounters_df = encounters_df[encounters_df["PatientPseudoKey"] != 2384]
    encounters_df = encounters_df[encounters_df["EncounterType"] == "Sykehuskontakt"]
    encounters_df["DischargeInstant"] = pd.to_datetime(encounters_df["DischargeInstant"], errors="coerce")

    # Clean and group DischargeDisposition
    drop_dispositions = {
        "*Unspecified",
        "Dratt på eget ansvar",
        "Feilregistrert",
        "Ikke ankommet akuttmottak",
        "Reserved by NUBC#70"
    }

    group_disposition_map = {
        "Som død - Melding går til sykepleietjeneste": "Som død",
        "Som død - Ingen melding går": "Som død",
        "Som død - Melding går til psykisk helsetjeneste": "Som død",
        "Ut til hjemmet - Ingen melding går": "Ut til hjemmet",
        "Ut til hjemmet - Melding går til psykisk helsetjeneste": "Ut til hjemmet",
        "Ut til hjemmet - Melding går til sykepleietjeneste": "Ut til hjemmet",
        "Ut til hjemmet (N/A)": "Ut til hjemmet"
    }

    if "DischargeDisposition" in encounters_df.columns:
        encounters_df = encounters_df[~encounters_df["DischargeDisposition"].isin(drop_dispositions)]
        encounters_df["DischargeDisposition"] = encounters_df["DischargeDisposition"].replace(group_disposition_map)

    # Group mappings for DischargeDestination (existing logic)
    group_hospital = {
        "Annen helseinstitusjon innen spesialisthelsetjenesten",
        "Annen helseinstitusjon innenfor spesialisthelsetjenesten",
        "Somatisk sykehus STO",
        "Psykiatrisk sykehus STO"
    }
    group_other = {
        "*Unspecified",
        "Annet",
        "Annen institusjon (ikke helse)"
    }

    def group_destination(dest):
        if pd.isna(dest):
            return None
        dest = dest.strip()
        if dest in group_hospital:
            return "Hospital"
        elif dest in group_other:
            return "Other/Unspecified"
        elif dest == "Kommunale institusjoner i HP":
            return "Sykehjem"
        else:
            return dest

    encounters_df["DischargeDestGroup"] = encounters_df["DischargeDestination"].apply(group_destination)

    # Prepare deciles from ADL values using equal-width bins
    adl_df["Decile"] = pd.cut(adl_df["Value"], 9, labels=False)
    bins = pd.cut(adl_df["Value"], 9)
    bin_labels = bins.astype(str)
    adl_df["DecileLabel"] = bin_labels

    # Load Øya_decisions.csv to identify long-term care decisions
    decisions_df = pd.read_csv("Øya_decisions.csv")
    decisions_df = decisions_df[
        (decisions_df["DecisionStatus"] == "Signert") &
        (decisions_df["DecisionTemplate"] == "Vedtak om langtidsopphold i institusjon")
    ]
    long_term_patients = set(decisions_df["PatientPseudoKey"])

    # Result structure
    unique_destinations = encounters_df["DischargeDestGroup"].dropna().replace({"Kommunale institusjoner i HP": "Sykehjem"}).unique().tolist()
    unique_destinations = [d for d in unique_destinations if d not in {"Other/Unspecified", "Annen (somatisk) enhet ved egen helseinstitusjon", "Som død"}]
    result_dict = defaultdict(lambda: {"Antall": 0, **{d: 0 for d in unique_destinations}, "Død, normal": 0, "Død, langtid": 0})

    date_diffs = []

    for _, encounter in encounters_df.iterrows():
        patient_id = encounter["PatientPseudoKey"]
        discharge_time = encounter["DischargeInstant"]
        dest_group = encounter["DischargeDestGroup"]

        # Custom logic for "Som død" to split by long-term decision
        if dest_group == "Som død":
            if patient_id in long_term_patients:
                dest_group = "Død, langtid"
            else:
                dest_group = "Død, normal"

        # Find closest ADL measurement for this patient relative to discharge time
        patient_adls = adl_df[adl_df["PatientPseudoKey"] == patient_id]
        if not accept_scores_after:
            patient_adls = patient_adls[patient_adls["MeasurementTime"] <= discharge_time]

        if not patient_adls.empty and pd.notna(dest_group) and dest_group not in {"Other/Unspecified", "Annen (somatisk) enhet ved egen helseinstitusjon"}:
            patient_adls = patient_adls.copy()
            patient_adls["TimeDiff"] = (patient_adls["MeasurementTime"] - discharge_time).abs()
            closest_adl = patient_adls.sort_values("TimeDiff").iloc[0]
            # Guard clause for null MeasurementTime
            if pd.isna(closest_adl["MeasurementTime"]):
                continue
            decile = closest_adl["DecileLabel"]
            result_dict[decile]["Antall"] += 1
            result_dict[decile][dest_group] += 1
            if pd.notna(closest_adl["MeasurementTime"]) and pd.notna(discharge_time):
                time_diff_days = abs((discharge_time - closest_adl["MeasurementTime"]).days)
                date_diffs.append(time_diff_days)

    if date_diffs:
        avg_days_diff = sum(date_diffs) / len(date_diffs)
        print(f"📏 Average days between latest {measurement_name} and discharge: {avg_days_diff:.2f} days")
    else:
        print("⚠️ No valid ADL-discharge matches to calculate average days difference.")

    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(result_dict, orient="index").fillna(0).reset_index()
    result_df.rename(columns={"index": "ADL_DecileLabel"}, inplace=True)
    result_df = result_df.sort_values(by="ADL_DecileLabel")

    # Add sum row
    sum_row = result_df.drop(columns=["ADL_DecileLabel"]).sum()
    sum_row["ADL_DecileLabel"] = "sum"
    result_df = pd.concat([result_df, pd.DataFrame([sum_row])], ignore_index=True)

    # Calculate and display percentages
    percentage_df = result_df.copy()
    numeric_cols = [col for col in percentage_df.columns if col not in ["ADL_DecileLabel", "Antall"]]
    for col in numeric_cols:
        percentage_df[col] = (percentage_df[col] / percentage_df["Antall"] * 100).round(1).astype(str) + '%'
    percentage_df = percentage_df.drop(columns=["Antall"])
    print("\n📊 Original counts by ADL decile:")
    print(result_df)
    print("\n📊 Percentage breakdown by ADL decile:")
    print(percentage_df)

    return result_df


# New function: analyze_hospital_outcomes_by_decile
def analyze_hospital_outcomes_by_decile(
    adl_path="Øya_2_ADL.csv",
    hospital_path="Øya_2_hospitalencounters.csv",
    measurement_name="R HP COCM IPLOS/ADL TOTAL VANLIG GJENNOMSNITT",
    accept_scores_after=True,
):
    # Load ADL
    adl_df = pd.read_csv(adl_path)
    adl_df = adl_df[adl_df["PatientPseudoKey"] != 2384]
    adl_df = adl_df[adl_df["MeasurementName"] == measurement_name]
    adl_df["MeasurementTime"] = pd.to_datetime(adl_df["MeasurementTime"], errors="coerce")
    adl_df = adl_df.dropna(subset=["MeasurementTime", "Value"])
    adl_df["Value"] = pd.to_numeric(adl_df["Value"], errors="coerce")
    adl_df = adl_df.dropna(subset=["Value"])

    # Load and preprocess hospital encounters
    df = pd.read_csv(hospital_path)
    df = df[df["PatientPseudoKey"] != 2384]
    df = df[df["AdmissionSource"] == "Bosted/arbeidsted"]
    df = df.dropna(subset=["DischargeDestination"])
    # Group mappings for merged categories
    merged_destination_map = {
        "Sykehjem": {"Sykehjem", "Langtidsopphold i sykehjem"}
        # "Tidsbegrenset opphold" group removed
    }
    # Create reverse map
    destination_reverse_map = {}
    for merged_label, originals in merged_destination_map.items():
        for original in originals:
            destination_reverse_map[original] = merged_label
    # Update DischargeDestination before grouping
    df["DischargeDestination"] = df["DischargeDestination"].replace(destination_reverse_map)
    # Then define the filtered valid set
    valid_destinations = {
        "Sykehjem",
        "Kommunale institusjoner i HP",
        "Som død",
        "Bosted/arbeidsted"
    }
    df = df[df["DischargeDestination"].isin(valid_destinations)]
    df = df[df["DischargeDestination"] != "Sykehjem"]
    df["EncounterEnd"] = pd.to_datetime(df["EncounterEnd"], errors="coerce")

    # --- Start Øya_decisions exclusion logic ---
    # Load and preprocess Øya_decisions.csv
    decisions_df = pd.read_csv("Øya_decisions.csv")
    decisions_df = decisions_df[
        (decisions_df["DecisionStatus"] == "Signert") &
        (decisions_df["DecisionTemplate"] == "Vedtak om langtidsopphold i institusjon")
    ]
    decisions_df["DecisionValidDate"] = pd.to_datetime(decisions_df["DecisionValidDate"], format="%d/%m/%Y", errors="coerce")
    # Parse hospital encounter dates as date-only for comparison
    df["EncounterEndDateOnly"] = pd.to_datetime(df["EncounterEnd"], errors="coerce").dt.date
    # Identify PatientPseudoKeys with valid long-term decisions before or on the day of encounter end
    decisions_df = decisions_df.dropna(subset=["DecisionValidDate", "PatientPseudoKey"])
    merged = pd.merge(
        df[["PatientPseudoKey", "EncounterEndDateOnly"]],
        decisions_df[["PatientPseudoKey", "DecisionValidDate"]],
        on="PatientPseudoKey",
        how="left"
    )
    merged = merged.dropna(subset=["DecisionValidDate"])
    merged["DecisionValidDate"] = pd.to_datetime(merged["DecisionValidDate"]).dt.date
    merged["FlagExclude"] = merged["DecisionValidDate"] <= merged["EncounterEndDateOnly"]
    patients_to_exclude = set(merged[merged["FlagExclude"]]["PatientPseudoKey"])
    df = df[~df["PatientPseudoKey"].isin(patients_to_exclude)]
    # Remove the temporary date-only column after filtering
    df = df.drop(columns=["EncounterEndDateOnly"])
    # --- End Øya_decisions exclusion logic ---

    # Note: Only DischargeDestGroup (from DischargeDestination) is used in this function, not DischargeDisposition.
    # If cleaning/grouping DischargeDisposition is needed, add similar logic as above.
    # For now, only group DischargeDestination as before.
    # No additional grouping needed: DischargeDestination has already been merged/grouped above.
    df["DischargeDestGroup"] = df["DischargeDestination"]  # Already grouped

    # Decile bucketing using equal-width bins
    adl_df["Decile"] = pd.cut(adl_df["Value"], 9, labels=False)
    bins = pd.cut(adl_df["Value"], 9)
    bin_labels = bins.astype(str)
    adl_df["DecileLabel"] = bin_labels

    # Result
    unique_destinations = df["DischargeDestGroup"].dropna().unique().tolist()
    result_dict = defaultdict(lambda: {"Antall": 0, **{d: 0 for d in unique_destinations}})

    date_diffs = []

    for _, enc in df.iterrows():
        patient_id = enc["PatientPseudoKey"]
        discharge_time = enc["EncounterEnd"]
        dest_group = enc["DischargeDestGroup"]

        patient_adls = adl_df[adl_df["PatientPseudoKey"] == patient_id]
        if not accept_scores_after:
            patient_adls = patient_adls[patient_adls["MeasurementTime"] <= discharge_time]

        if not patient_adls.empty and pd.notna(dest_group):
            patient_adls = patient_adls.copy()
            patient_adls["TimeDiff"] = (patient_adls["MeasurementTime"] - discharge_time).abs()
            closest_adl = patient_adls.sort_values("TimeDiff").iloc[0]

            if pd.isna(closest_adl["MeasurementTime"]):
                continue

            decile = closest_adl["DecileLabel"]
            result_dict[decile]["Antall"] += 1
            result_dict[decile][dest_group] += 1

            if pd.notna(discharge_time):
                time_diff_days = abs((discharge_time - closest_adl["MeasurementTime"]).days)
                date_diffs.append(time_diff_days)

    if date_diffs:
        avg_days_diff = sum(date_diffs) / len(date_diffs)
        print(f"📏 Average days between latest {measurement_name} and hospital discharge: {avg_days_diff:.2f} days")
    else:
        print("⚠️ No valid ADL-hospital matches to calculate average days difference.")

    result_df = pd.DataFrame.from_dict(result_dict, orient="index").fillna(0).reset_index()
    result_df.rename(columns={"index": "ADL_DecileLabel"}, inplace=True)
    result_df = result_df.sort_values(by="ADL_DecileLabel")

    # Add sum row
    sum_row = result_df.drop(columns=["ADL_DecileLabel"]).sum()
    sum_row["ADL_DecileLabel"] = "sum"
    result_df = pd.concat([result_df, pd.DataFrame([sum_row])], ignore_index=True)

    # Calculate and display percentages
    percentage_df = result_df.copy()
    numeric_cols = [col for col in percentage_df.columns if col not in ["ADL_DecileLabel", "Antall"]]
    for col in numeric_cols:
        percentage_df[col] = (percentage_df[col] / percentage_df["Antall"] * 100).round(1).astype(str) + '%'
    percentage_df = percentage_df.drop(columns=["Antall"])
    print("\n📊 Original counts by ADL decile:")
    print(result_df)
    print("\n📊 Percentage breakdown by ADL decile:")
    print(percentage_df)

    return result_df

if __name__ == "__main__":
    
    #summary = analyze_cfs_outcomes()
    #print(summary)

    #interval_summary = analyze_cfs_destinations_by_intervals()
    #print(interval_summary)

    #plot_cfs_vs_adl()
    
    analyze_adl_outcomes_by_decile()
    analyze_hospital_outcomes_by_decile()
    # Uncomment the following line to save the summary to a CSV files