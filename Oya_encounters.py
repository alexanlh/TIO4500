import pandas as pd
import plotly.graph_objects as go

### --- Reusable Helper Functions --- ###

def load_and_filter_encounters(file_path):
    df = pd.read_csv(file_path)
    df = df[df["PatientPseudoKey"] != 2384]  # Remove test patient
    df = df[df["EncounterType"] == "Sykehuskontakt"]  # Only sykehuskontakt
    df["LengthOfStay"] = pd.to_numeric(df["LengthOfStay"], errors="coerce")
    return df

def remove_zero_length_stays(df):
    return df[df["LengthOfStay"] > 0]

def get_revisiting_patients(df):
    visit_counts = df["PatientPseudoKey"].value_counts()
    return visit_counts[visit_counts > 1].index

def get_last_encounter_per_patient(df):
    idx_last = df.groupby("PatientPseudoKey")["EncounterPseudoKey"].idxmax()
    return df.loc[idx_last]

### --- Analysis Functions --- ###

def analyze_encounters(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)

    total_visits = df.groupby("Department")["PatientPseudoKey"].count().reset_index()
    total_visits.columns = ["Department", "TotalVisits"]

    unique_visits = df[["PatientPseudoKey", "Department"]].drop_duplicates()
    unique_patients = unique_visits.groupby("Department")["PatientPseudoKey"].nunique().reset_index()
    unique_patients.columns = ["Department", "UniquePatientCount"]

    avg_los = df.groupby("Department")["LengthOfStay"].mean().reset_index()
    avg_los.columns = ["Department", "AverageLengthOfStay"]

    disposition_counts = df.groupby(["Department", "DischargeDisposition"]).size().unstack(fill_value=0).reset_index()
    disposition_counts["Deaths"] = disposition_counts.get("Som d\u00f8d - Ingen melding g\u00e5r", 0)
    disposition_counts["SentHome"] = disposition_counts.get("Ut til hjemmet - Ingen melding g\u00e5r", 0)
    disposition_counts["SentToInstitution"] = disposition_counts.get("Til annen enhet - Ingen melding g\u00e5r", 0)
    disposition_summary = disposition_counts[["Department", "Deaths", "SentHome", "SentToInstitution"]]

    summary = pd.merge(total_visits, unique_patients, on="Department")
    summary = pd.merge(summary, avg_los, on="Department")
    summary = pd.merge(summary, disposition_summary, on="Department")

    return summary.sort_values(by="Department").reset_index(drop=True)

def analyze_dispositions_by_service(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)

    disposition_map = {
        "Som d\u00f8d - Ingen melding g\u00e5r": "Deaths",
        "Ut til hjemmet - Ingen melding g\u00e5r": "SentHome",
        "Til annen enhet - Ingen melding g\u00e5r": "SentToInstitution"
    }

    df = df[df["DischargeDisposition"].isin(disposition_map.keys())]
    df["DispositionCategory"] = df["DischargeDisposition"].map(disposition_map)

    unique_cases = df[["PatientPseudoKey", "HospitalService", "DispositionCategory"]].drop_duplicates()
    summary = unique_cases.groupby(["HospitalService", "DispositionCategory"]).size().unstack(fill_value=0).reset_index()

    for col in ["Deaths", "SentHome", "SentToInstitution"]:
        if col not in summary:
            summary[col] = 0

    return summary.sort_values(by="Deaths", ascending=False).reset_index(drop=True)

def count_admissions_by_source(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)
    return df["AdmissionSource"].value_counts(dropna=False).reset_index().rename(columns={"index": "AdmissionSource", "AdmissionSource": "NumberOfEncounters"})

def get_revisit_details(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)
    df = remove_zero_length_stays(df)

    revisiting_patients = get_revisiting_patients(df)
    df = df[df["PatientPseudoKey"].isin(revisiting_patients)]

    cols = ["PatientPseudoKey", "EncounterPseudoKey", "EncounterStart", "EncounterEnd", "LengthOfStay", "AdmittingDepartment", "AdmissionSource", "DischargeDisposition"]
    return df[cols].sort_values(by=["PatientPseudoKey", "EncounterStart"]).reset_index(drop=True)

def count_revisits(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)
    df = remove_zero_length_stays(df)

    visit_counts = df["PatientPseudoKey"].value_counts().reset_index()
    visit_counts.columns = ["PatientPseudoKey", "VisitCount"]
    revisit_counts = visit_counts[visit_counts["VisitCount"] > 1].reset_index(drop=True)

    avg_visits = revisit_counts["VisitCount"].mean()
    print(f"\U0001F4CA Average number of visits among revisiting patients (LengthOfStay > 0): {avg_visits:.2f}\n")

    return revisit_counts

def count_last_dispositions_from_revisit_list(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)
    df = remove_zero_length_stays(df)

    revisiting_patients = get_revisiting_patients(df)
    df = df[df["PatientPseudoKey"].isin(revisiting_patients)]
    last_encounters = get_last_encounter_per_patient(df)

    disposition_counts = last_encounters["DischargeDisposition"].value_counts(dropna=False).reset_index()
    disposition_counts.columns = ["ListOfDischargeDisposition", "NumberOfLastOccurences"]

    return disposition_counts

def inflow_analysis(file_path="\u00d8ya_encounters.csv", output_file="WriteToExcel.xlsx"):
    df = load_and_filter_encounters(file_path)
    inflow_table = df.groupby(["Department", "AdmissionSource"]).size().unstack(fill_value=0).reset_index()

    outflow_table = df.groupby(["Department", "DischargeDestination"]).size().unstack(fill_value=0).reset_index()

    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        inflow_table.to_excel(writer, index=False, sheet_name="Inflow")
        outflow_table.to_excel(writer, index=False, sheet_name="Outflow")

    print(f"\u2705 Inflow and Outflow tables written to {output_file}")

def flow_visualization(file_path="\u00d8ya_encounters.csv"):
    df = load_and_filter_encounters(file_path)
    df = df.dropna(subset=["AdmissionSource", "DischargeDestination"])

    group_other = ["*Unspecified", "Annet", "Annen institusjon (ikke helse)"]
    group_hospital = ["Annen helseinstitusjon innenfor spesialisthelsetjenesten", "Somatisk sykehus STO", "Psykiatrisk sykehus STO"]

    df["SourceGroup"] = df["AdmissionSource"].replace({**{v: "Other/Unspecified" for v in group_other}, **{v: "Hospital (SpecHelsetjeneste)" for v in group_hospital}})
    df["SourceGroup"] = df["SourceGroup"].fillna(df["AdmissionSource"])

    df["DestGroup"] = df["DischargeDestination"].replace({**{v: "Other/Unspecified" for v in group_other}, **{v: "Hospital (SpecHelsetjeneste)" for v in group_hospital}})
    df["DestGroup"] = df["DestGroup"].fillna(df["DischargeDestination"])

    df["SourceLabel"] = "From: " + df["SourceGroup"]
    df["DestLabel"] = "To: " + df["DestGroup"]

    source_nodes = df["SourceLabel"].unique().tolist()
    dept_nodes = df["Department"].unique().tolist()
    dest_nodes = df["DestLabel"].unique().tolist()
    all_nodes = source_nodes + dept_nodes + dest_nodes
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    links = []
    for _, row in df.iterrows():
        links.append((node_indices[row["SourceLabel"]], node_indices[row["Department"]]))
        links.append((node_indices[row["Department"]], node_indices[row["DestLabel"]]))

    from collections import Counter
    link_counts = Counter(links)

    sankey_links = {
        "source": [src for (src, tgt), _ in link_counts.items()],
        "target": [tgt for (src, tgt), _ in link_counts.items()],
        "value": [count for (_, _), count in link_counts.items()]
    }

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes
        ),
        link=sankey_links
    )])

    fig.update_layout(title_text="Pasientflyt: Inngang → Avdeling → Utskrivning", font_size=10)
    fig.show()

def flow_visualization_simplified_with_colors(file_path="Øya_encounters.csv"):
    df = pd.read_csv(file_path)
    df = df[df["PatientPseudoKey"] != 2384]  # Exclude test patient
    df = df[df["EncounterType"] == "Sykehuskontakt"]
    df = df.dropna(subset=["AdmissionSource", "DischargeDestination"])

    # Strip and normalize
    df["AdmissionSource"] = df["AdmissionSource"].astype(str).str.strip()
    df["DischargeDestination"] = df["DischargeDestination"].astype(str).str.strip()
    df["Department"] = df["Department"].astype(str).str.strip()

    # Define grouping
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

    # Map sources
    df["SourceGroup"] = df["AdmissionSource"].replace({
        **{v: "Hospital" for v in group_hospital},
        **{v: "Other/Unspecified" for v in group_other}
    }).fillna(df["AdmissionSource"]).apply(lambda x: "From: " + x)

    # Map destinations
    df["DestGroup"] = df["DischargeDestination"].replace({
        **{v: "Hospital" for v in group_hospital},
        **{v: "Other/Unspecified" for v in group_other}
    }).fillna(df["DischargeDestination"]).apply(lambda x: "To: " + x)

    # Simplify department names
    short_stay = {
        "TRD H ØYA HELSEHUS 4. ET. AVD. B",
        "TRD H ØYA HELSEHUS 5. ET. AVD. A",
        "TRD H ØYA HELSEHUS 6. ET. AVD. A",
        "TRD H ØYA HELSEHUS 6. ET. AVD. B"
    }

    df["SimplifiedDept"] = df["Department"].replace({
        "TRD H ØYA HELSEHUS 5. ET. AVD. B": "Palliative",
        "TRD H ØYA HELSEHUS 3. ET. AVD. A": "Longterm",
        **{d: "ShortStay" for d in short_stay}
    })

    # Build node list
    source_nodes = df["SourceGroup"].unique().tolist()
    dept_nodes = df["SimplifiedDept"].unique().tolist()
    dest_nodes = df["DestGroup"].unique().tolist()
    all_nodes = source_nodes + dept_nodes + dest_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    # Build links
    links = []
    inflow = df.groupby(["SourceGroup", "SimplifiedDept"]).size().reset_index(name="count")
    for _, row in inflow.iterrows():
        links.append(dict(
            source=node_idx[row["SourceGroup"]],
            target=node_idx[row["SimplifiedDept"]],
            value=row["count"]
        ))

    outflow = df.groupby(["SimplifiedDept", "DestGroup"]).size().reset_index(name="count")
    for _, row in outflow.iterrows():
        links.append(dict(
            source=node_idx[row["SimplifiedDept"]],
            target=node_idx[row["DestGroup"]],
            value=row["count"]
        ))

    # Define colors by type
    color_map = {}
    for node in all_nodes:
        if node.startswith("From:"):
            color_map[node] = "cornflowerblue"
        elif node.startswith("To:"):
            color_map[node] = "lightgreen"
        elif node == "Palliative":
            color_map[node] = "indianred"
        elif node == "Longterm":
            color_map[node] = "slategray"
        elif node == "ShortStay":
            color_map[node] = "orange"
        else:
            color_map[node] = "lightgray"

    # Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_nodes,
            color=[color_map[n] for n in all_nodes]
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links]
        )
    ))

    fig.update_layout(title="Pasientflyt (Forenklet, Fargekodet)", font_size=11)
    fig.show()

def one_unit_visualization(file_path="Øya_encounters.csv"):
    # Load and filter
    df = pd.read_csv(file_path)
    df = df[df["PatientPseudoKey"] != 2384]
    df = df[df["EncounterType"] == "Sykehuskontakt"]
    df = df.dropna(subset=["AdmissionSource", "DischargeDestination"])

    # Normalize text
    df["AdmissionSource"] = df["AdmissionSource"].astype(str).str.strip()
    df["DischargeDestination"] = df["DischargeDestination"].astype(str).str.strip()

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

    # Inflow and outflow grouping
    df["SourceGroup"] = df["AdmissionSource"].replace({
        **{v: "Hospital" for v in group_hospital},
        **{v: "Other/Unspecified" for v in group_other}
    }).fillna(df["AdmissionSource"]).apply(lambda x: "From: " + x)

    df["DestGroup"] = df["DischargeDestination"].replace({
        **{v: "Hospital" for v in group_hospital},
        **{v: "Other/Unspecified" for v in group_other}
    }).fillna(df["DischargeDestination"]).apply(lambda x: "To: " + x)

    df["Unit"] = "Øya helsehus"

    # Build nodes
    source_nodes = df["SourceGroup"].unique().tolist()
    unit_node = ["Øya helsehus"]
    dest_nodes = df["DestGroup"].unique().tolist()
    all_nodes = source_nodes + unit_node + dest_nodes
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    # Build links
    links = []
    inflow = df.groupby("SourceGroup").size().reset_index(name="count")
    for _, row in inflow.iterrows():
        links.append(dict(
            source=node_idx[row["SourceGroup"]],
            target=node_idx["Øya helsehus"],
            value=row["count"]
        ))

    outflow = df.groupby("DestGroup").size().reset_index(name="count")
    for _, row in outflow.iterrows():
        links.append(dict(
            source=node_idx["Øya helsehus"],
            target=node_idx[row["DestGroup"]],
            value=row["count"]
        ))

    # Color coding
    color_map = {}
    for node in all_nodes:
        if node.startswith("From:"):
            color_map[node] = "cornflowerblue"
        elif node.startswith("To:"):
            color_map[node] = "lightgreen"
        elif node == "Øya helsehus":
            color_map[node] = "darkorange"
        else:
            color_map[node] = "lightgray"

    # Sankey Diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_nodes,
            color=[color_map[n] for n in all_nodes]
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links]
        )
    ))

    fig.update_layout(title="Pasientflyt – Hele Øya helsehus som én enhet", font_size=11)
    fig.show()

if __name__ == "__main__":
    print(analyze_encounters())
    print(analyze_dispositions_by_service())
    print(count_admissions_by_source())
    print(get_revisit_details())
    print(count_revisits())
    print(count_last_dispositions_from_revisit_list())
    inflow_analysis()
    flow_visualization()
    flow_visualization_simplified_with_colors()
    one_unit_visualization()

