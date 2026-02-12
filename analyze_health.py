import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set plotting style
sns.set_theme(style="white")

DATA_DIR = "my_whoop_data_2026_02_11"

def load_data():
    print("Loading data...")
    try:
        journal = pd.read_csv(os.path.join(DATA_DIR, "journal_entries.csv"))
        phys = pd.read_csv(os.path.join(DATA_DIR, "physiological_cycles.csv"))
        sleeps = pd.read_csv(os.path.join(DATA_DIR, "sleeps.csv"))
        workouts = pd.read_csv(os.path.join(DATA_DIR, "workouts.csv"))
        happiness = pd.read_csv(os.path.join(DATA_DIR, "subjective_happiness.csv"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None
    return journal, phys, sleeps, workouts, happiness

def process_journal(journal_df):
    print("Processing Journal Entries...")
    # Convert index to datetime for merging
    journal_df['Cycle start time'] = pd.to_datetime(journal_df['Cycle start time'])

    # Drop duplicates if any (same cycle, same question)
    journal_df = journal_df.drop_duplicates(subset=['Cycle start time', 'Question text'])

    # Pivot: Index=Cycle start time, Columns=Question text, Values=Answered yes
    # "Answered yes" seems to be boolean or string 'true'/'false'.
    # Inspect unique values if needed, but safe conversion:
    journal_df['Answered yes'] = journal_df['Answered yes'].astype(str).str.lower().map({'true': 1, 'false': 0})

    pivoted = journal_df.pivot(index='Cycle start time', columns='Question text', values='Answered yes')
    return pivoted

def process_physiological(phys_df):
    print("Processing Physiological Cycles...")
    phys_df['Cycle start time'] = pd.to_datetime(phys_df['Cycle start time'])
    phys_df['Wake onset'] = pd.to_datetime(phys_df['Wake onset'])

    # Create Date from Wake onset (Day N)
    phys_df['Date'] = phys_df['Wake onset'].dt.normalize()

    # Select columns
    cols = ['Cycle start time', 'Date', 'Recovery score %', 'Resting heart rate (bpm)', 'Heart rate variability (ms)', 'Day Strain']
    # Filter only columns that exist
    cols = [c for c in cols if c in phys_df.columns]
    return phys_df[cols]

def process_sleeps(sleeps_df):
    print("Processing Sleeps...")
    sleeps_df['Cycle start time'] = pd.to_datetime(sleeps_df['Cycle start time'])
    cols = ['Cycle start time', 'Sleep efficiency %', 'Sleep performance %']
    return sleeps_df[cols]

def process_workouts(workouts_df):
    print("Processing Workouts...")
    workouts_df['Workout start time'] = pd.to_datetime(workouts_df['Workout start time'])
    workouts_df['Workout Date'] = workouts_df['Workout start time'].dt.normalize()

    # Calculate Zone Minutes
    for i in range(1, 6):
        col_pct = f'HR Zone {i} %'
        col_min = f'Zone {i} Minutes'
        if col_pct in workouts_df.columns:
            workouts_df[col_min] = workouts_df['Duration (min)'] * (workouts_df[col_pct] / 100.0)

    # Aggregate by Date
    agg_funcs = {
        'Activity Strain': 'sum',
        'Duration (min)': 'sum'
    }
    # Add zone columns to aggregation
    for i in range(1, 6):
        col_min = f'Zone {i} Minutes'
        if col_min in workouts_df.columns:
            agg_funcs[col_min] = 'sum'

    aggregated = workouts_df.groupby('Workout Date').agg(agg_funcs).reset_index()

    # Shift Date: Workout Date (Day N-1) -> Affects Recovery Date (Day N)
    # So we want to merge Workout Date X with Recovery Date X+1.
    # Create 'Date' column for merging = Workout Date + 1 Day
    aggregated['Date'] = aggregated['Workout Date'] + pd.Timedelta(days=1)

    return aggregated

def process_happiness(happiness_df):
    print("Processing Happiness...")
    happiness_df['Date'] = pd.to_datetime(happiness_df['Date']).dt.normalize()
    return happiness_df

def main():
    journal, phys, sleeps, workouts, happiness = load_data()
    if journal is None:
        return

    # 1. Process Main Data (Phys + Sleep + Journal)
    phys_proc = process_physiological(phys)
    sleeps_proc = process_sleeps(sleeps)
    journal_proc = process_journal(journal)

    # Merge Phys and Sleep
    main_df = pd.merge(phys_proc, sleeps_proc, on='Cycle start time', how='left')

    # Merge Journal (on Cycle start time)
    # Journal index is Cycle start time
    main_df = pd.merge(main_df, journal_proc, left_on='Cycle start time', right_index=True, how='left')

    # 2. Process Workouts
    workouts_proc = process_workouts(workouts)

    # 3. Process Happiness
    happiness_proc = process_happiness(happiness)

    # 4. Final Merge on Date
    # Merge Workouts (Date is already shifted to match Recovery Day)
    full_df = pd.merge(main_df, workouts_proc, on='Date', how='left')

    # Fill NaN for workout columns with 0 (rest days)
    workout_cols = ['Activity Strain', 'Duration (min)'] + [c for c in workouts_proc.columns if 'Zone' in c and 'Minutes' in c]
    # Only fill columns that exist in full_df
    workout_cols = [c for c in workout_cols if c in full_df.columns]
    full_df[workout_cols] = full_df[workout_cols].fillna(0)
    print(f"Filled NaN values with 0 for {len(workout_cols)} workout columns (rest days).")

    # Merge Happiness
    full_df = pd.merge(full_df, happiness_proc, on='Date', how='left')

    print(f"Final dataset shape: {full_df.shape}")

    # 5. Correlation Analysis
    # Drop non-numeric columns for correlation
    numeric_df = full_df.select_dtypes(include=['number'])

    # Drop ID columns or meaningless columns if any
    # 'Cycle start time' and 'Date' are not numeric, so they are dropped automatically or we should drop them if they converted to numbers (timestamps).
    # Check if 'Day Strain' is numeric.

    corr_matrix = numeric_df.corr()

    target = 'Recovery score %'
    if target in corr_matrix.columns:
        print(f"\n--- Top 5 Positive Influences on {target} ---")
        # Sort descending, exclude target itself
        correlations = corr_matrix[target].drop(target)
        print(correlations.sort_values(ascending=False).head(5))

        print(f"\n--- Top 5 Negative Influences on {target} ---")
        print(correlations.sort_values(ascending=True).head(5))
    else:
        print(f"Target {target} not found in correlation matrix.")

    # 6. Heatmap
    plt.figure(figsize=(20, 16))
    # Filter to show only relevant rows/cols if too large?
    # For now, show all.
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix: Behaviors, Workouts, and Recovery")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("\nHeatmap saved to correlation_heatmap.png")

if __name__ == "__main__":
    main()
