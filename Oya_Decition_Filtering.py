import pandas as pd
from Oya_encounters import load_and_filter_encounters, get_last_encounter_per_patient  
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


def load_and_filter_decisions(file_path="\u00d8ya_decisions.csv"):
    # Load CSV
    df = pd.read_csv(file_path)

    # Remove test patient
    df = df[df["PatientPseudoKey"] != 2384]

    # Keep only relevant DecisionTemplate values
    valid_templates = [
        "Vedtak om helsetjenester i hjemmet - Tjenester i hjemmet",
        "Vedtak om tidsbegrenset opphold",
        "Vedtak om langtidsopphold i institusjon",
        "Vedtak om praktisk bistand daglige gj\u00f8rem\u00e5l - Tjenester i hjemmet"
    ]
    df = df[df["DecisionTemplate"].isin(valid_templates)]

    # Keep only valid DecisionStatus values
    valid_statuses = ["Signert", "Omgjort", "Inaktiv"]
    df = df[df["DecisionStatus"].isin(valid_statuses)]

    return df

def analyze_decision_patterns(file_path="Øya_decisions.csv", min_support=0.1):
    # Load data
    df = pd.read_csv(file_path)

    # Step 1: Filter out test patient
    df = df[df["PatientPseudoKey"] != 2384]

    # Step 2: Keep only valid templates
    valid_templates = [
        "Vedtak om helsetjenester i hjemmet - Tjenester i hjemmet",
        "Vedtak om tidsbegrenset opphold",
        "Vedtak om langtidsopphold i institusjon",
        "Vedtak om praktisk bistand daglige gjøremål - Tjenester i hjemmet"
    ]
    df = df[df["DecisionTemplate"].isin(valid_templates)]

    # Step 3: Keep only valid statuses
    valid_statuses = ["Signert", "Omgjort", "Inaktiv"]
    df = df[df["DecisionStatus"].isin(valid_statuses)]

    # Step 4: Group by PatientPseudoKey to get unique decision templates per patient
    transactions = (
        df.groupby("PatientPseudoKey")["DecisionTemplate"]
        .apply(lambda x: list(set(x)))  # Use set() to remove duplicates
        .tolist()
    )

    # Step 5: TransactionEncoder
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # Step 6: Run FP-Growth
    frequent_itemsets = fpgrowth(te_df, min_support=min_support, use_colnames=True)

    # Optional: Add length column for readability
    frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)

    return frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

def analyze_all_templates_for_valid_patients(file_path="Øya_decisions.csv", min_support=0.1, export_excel=True):
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth

    # Load and filter data
    df = pd.read_csv(file_path)
    df = df[df["PatientPseudoKey"] != 2384]

    valid_templates = [
        "Vedtak om helsetjenester i hjemmet - Tjenester i hjemmet",
        "Vedtak om tidsbegrenset opphold",
        "Vedtak om langtidsopphold i institusjon",
        "Vedtak om praktisk bistand daglige gjøremål - Tjenester i hjemmet"
    ]
    valid_statuses = ["Signert", "Omgjort", "Inaktiv"]
    df = df[df["DecisionStatus"].isin(valid_statuses)]

    # Get patients who have at least one valid template
    valid_patients = df[df["DecisionTemplate"].isin(valid_templates)]["PatientPseudoKey"].unique()

    # Keep all rows for those patients, even if template is not valid
    df = df[df["PatientPseudoKey"].isin(valid_patients)]

    # Group DecisionTemplate per patient
    transactions = (
        df.groupby("PatientPseudoKey")["DecisionTemplate"]
        .apply(lambda x: list(set(x)))  # remove duplicates within patient
        .tolist()
    )

    # Transaction encoding and fpgrowth
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = fpgrowth(te_df, min_support=min_support, use_colnames=True)
    frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets_sorted = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    # Export
    if export_excel:
        frequent_itemsets_sorted.to_excel("fpgrowth_all_templates.xlsx", index=False)
        print("✅ Results written to fpgrowth_all_templates.xlsx")
    else:
        frequent_itemsets_sorted.to_csv("fpgrowth_all_templates.csv", index=False)
        print("✅ Results written to fpgrowth_all_templates.csv")

    return frequent_itemsets_sorted

def analyze_outcomes_for_longterm_decision(decisions_path="\u00d8ya_decisions.csv", encounters_path="\u00d8ya_encounters.csv"):
    # Step 1: Load decisions
    decisions = pd.read_csv(decisions_path)

    # Remove test patient
    decisions = decisions[decisions["PatientPseudoKey"] != 2384]

    # Only include valid statuses
    valid_statuses = ["Signert", "Omgjort", "Inaktiv"]
    decisions = decisions[decisions["DecisionStatus"].isin(valid_statuses)]

    # Step 2: Identify patients with a long-term institution decision
    longterm_patients = decisions[
        decisions["DecisionTemplate"] == "Vedtak om langtidsopphold i institusjon"
    ]["PatientPseudoKey"].unique()

    # Step 3: Load and filter encounters
    encounters = load_and_filter_encounters(encounters_path)

    # Step 4: Keep only patients who were granted long-term institution stay
    encounters = encounters[encounters["PatientPseudoKey"].isin(longterm_patients)]

    # Step 5: Get last encounter per patient
    last_encounters = get_last_encounter_per_patient(encounters)

    # Step 6: Count DischargeDisposition for these patients
    disposition_counts = (
        last_encounters["DischargeDisposition"]
        .value_counts(dropna=False)
        .reset_index()
    )
    disposition_counts.columns = ["DischargeDisposition", "NumberOfPatients"]

    # Step 7: For those with "Til annen enhet - Ingen melding g\u00e5r", count DischargeDestination
    to_other_institution = last_encounters[
        last_encounters["DischargeDisposition"] == "Til annen enhet - Ingen melding g\u00e5r"
    ]
    discharge_destination_counts = (
        to_other_institution["DischargeDestination"]
        .value_counts(dropna=False)
        .reset_index()
    )
    discharge_destination_counts.columns = ["DischargeDestination", "Count"]

    print("\nBreakdown of DischargeDestination for patients sent to another unit:")
    print(discharge_destination_counts)

    # Step 8: Analyze "Ut til hjemmet - Ingen melding g\u00e5r" with vedtak check
    sent_home = last_encounters[
        last_encounters["DischargeDisposition"] == "Ut til hjemmet - Ingen melding g\u00e5r"
    ][["PatientPseudoKey", "DischargeInstant"]].copy()

    # Parse DischargeInstant to date (dd/mm/YYYY)
    sent_home["DischargeInstant"] = pd.to_datetime(sent_home["DischargeInstant"], format="%d/%m/%Y", errors="coerce")

    # Extract valid decisions for these patients
    relevant_decisions = decisions[
        decisions["PatientPseudoKey"].isin(sent_home["PatientPseudoKey"])
    ][["PatientPseudoKey", "DecisionValidDate"]].copy()
    relevant_decisions["DecisionValidDate"] = pd.to_datetime(relevant_decisions["DecisionValidDate"], format="%d/%m/%Y %H:%M", errors="coerce")

    # Keep only earliest valid date per patient
    earliest_decision = relevant_decisions.groupby("PatientPseudoKey")["DecisionValidDate"].min().reset_index()

    # Merge with sent_home patients
    sent_home = sent_home.merge(earliest_decision, on="PatientPseudoKey", how="left")

    # Compare date parts only
    sent_home["Category"] = sent_home.apply(
        lambda row: "Sendt hjem med vedtak" if pd.notnull(row["DecisionValidDate"]) and row["DecisionValidDate"].date() <= row["DischargeInstant"].date() else "Sendt hjem uten vedtak",
        axis=1
    )

    category_counts = sent_home["Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "NumberOfPatients"]

    print("\nBreakdown of patients sent home:")
    print(category_counts)

    return disposition_counts


if __name__ == "__main__":
    decisions = load_and_filter_decisions()
    print(decisions.head())
    patterns = analyze_decision_patterns()
    print(patterns)
    #patterns_all = analyze_all_templates_for_valid_patients()
    #print(patterns_all)
    result = analyze_outcomes_for_longterm_decision()
    print("\nDischargeDisposition summary:")
    print(result)
