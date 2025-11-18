import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from io import BytesIO
from typing import Optional, Dict, List
import tempfile

# Constants from the original script - UPDATED FOR EXTENDED ZONES
KM_LIMIT = 125.0         # hard max km per schedule
MAX_HOURS = 12.0         # max duration first pickup -> last drop (hours)
MAX_ZONE_DEPTH = 4       # 0,1,2,3,4 hops allowed
GAP_RULES = {            # Time gaps by distance
    0: 10,  # same zone
    1: 10,  # direct neighbor
    2: 15,  # 2-hop
    3: 20,  # 3-hop
    4: 25   # 4-hop
}

# List of possible run number column names (for dynamic detection)
POSSIBLE_RUN_COLS = [
    "TTM Number", "TTM_Number", "TTM number", "ttm number",
    "Run Number", "Run_Number", "Run number", "run number",
    "Run ID", "run_id", "ID", "id"
]

# Adapted functions from the original script
# (load_trips and load_zone_graph now accept file-like objects instead of paths)

def parse_time_str(s: str) -> datetime:
    s = str(s).strip()
    # Dummy date + time; only time-of-day matters for differences
    return datetime.strptime(s, "%H:%M:%S")

def load_trips(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df["pickup_dt"] = df["First Pickup Time"].apply(parse_time_str)
    df["drop_dt"] = df["Last Dropoff Time"].apply(parse_time_str)
    df = df.sort_values("pickup_dt").reset_index(drop=True)
    return df

def load_zone_graph(uploaded_file) -> dict:
    # Determine format from uploaded file name or content
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        zdf = pd.read_csv(uploaded_file)
    else:
        zdf = pd.read_excel(uploaded_file)

    neighbors = defaultdict(set)
    for _, row in zdf.iterrows():
        if pd.isna(row["Primary Zone"]):
            continue
        p = int(row["Primary Zone"])
        neighbors[p].add(p)
        raw = "" if pd.isna(row.get("Backup Zones")) else str(row["Backup Zones"])
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                b = int(part)
                neighbors[p].add(b)
                neighbors[b].add(p)  # symmetric
    return neighbors

def zone_distance(neighbors: dict, start: int, target: int, max_depth: int = 4):
    """BFS up to max_depth; return distance (0..max_depth) or None if >max_depth/unreachable."""
    start = int(start)
    target = int(target)
    if start == target:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        z, d = q.popleft()
        if d == max_depth:
            continue
        for nb in neighbors.get(z, []):
            if nb in visited:
                continue
            nd = d + 1
            if nb == target:
                return nd
            visited.add(nb)
            q.append((nb, nd))
    return None

def build_schedules(trips: pd.DataFrame, neighbors: dict):
    UNASSIGNED = set(trips.index.tolist())
    schedules = []
    schedule_counter = 1

    while UNASSIGNED:
        # Start with earliest unassigned pickup
        earliest_idx = min(UNASSIGNED, key=lambda i: trips.loc[i, "pickup_dt"])
        current_indices = []
        total_km = 0.0
        current_idx = earliest_idx
        first_pickup = trips.loc[current_idx, "pickup_dt"]

        while True:
            current_indices.append(current_idx)
            UNASSIGNED.remove(current_idx)
            total_km += float(trips.loc[current_idx, "KM"])

            # Stop if km cap reached
            if total_km >= KM_LIMIT - 1e-6:
                break

            prev_drop_time = trips.loc[current_idx, "drop_dt"]
            prev_drop_zone = int(trips.loc[current_idx, "Last Dropoff Zone"])

            min_pickup_time_base = prev_drop_time
            good_candidates = []

            for i in UNASSIGNED:
                pick_zone = int(trips.loc[i, "First Pickup Zone"])
                # Zone distance check
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone, max_depth=MAX_ZONE_DEPTH)
                if dist is None or dist > MAX_ZONE_DEPTH:
                    continue

                # Time gap rule based on distance
                if dist in GAP_RULES:
                    min_gap = GAP_RULES[dist]
                else:
                    continue  # Should not happen, but safety

                min_pickup_time = min_pickup_time_base + timedelta(minutes=min_gap)
                pick_time = trips.loc[i, "pickup_dt"]
                if pick_time < min_pickup_time:
                    continue

                # 12-hour limit check if we add this trip
                candidate_drop = trips.loc[i, "drop_dt"]
                duration_hours = (candidate_drop - first_pickup).total_seconds() / 3600.0
                if duration_hours > MAX_HOURS + 1e-6:
                    continue

                # KM limit check
                candidate_km = float(trips.loc[i, "KM"])
                if total_km + candidate_km > KM_LIMIT + 1e-6:
                    continue

                good_candidates.append((i, pick_time))

            if not good_candidates:
                break

            # Choose earliest pickup among candidates
            good_candidates.sort(key=lambda t: t[1])
            current_idx = good_candidates[0][0]

        schedules.append(
            {
                "id": f"SCH-{schedule_counter:03d}",
                "trip_indices": current_indices,
            }
        )
        schedule_counter += 1

    return schedules

def build_schedules_from_unassigned(trips: pd.DataFrame, neighbors: dict, unassigned_indices: List[int]) -> List[Dict]:
    """Variant of build_schedules, but only on given indices."""
    # Subset trips to unassigned
    sub_trips = trips.loc[unassigned_indices].copy().reset_index(drop=True)
    sub_trips.index = unassigned_indices  # Preserve original indices
    return build_schedules(sub_trips, neighbors)  # Reuse your func

def build_summary(trips: pd.DataFrame, schedules: list) -> pd.DataFrame:
    rows = []
    for s in schedules:
        idxs = s["trip_indices"]
        kmtotal = sum(float(trips.loc[i, "KM"]) for i in idxs)
        pickup_times = [trips.loc[i, "pickup_dt"] for i in idxs]
        drop_times = [trips.loc[i, "drop_dt"] for i in idxs]
        rows.append(
            {
                "Schedule_ID": s["id"],
                "Trip_Count": len(idxs),
                "Total_KM": round(kmtotal, 3),
                "Start_Time": min(pickup_times).strftime("%H:%M"),
                "End_Time": max(drop_times).strftime("%H:%M"),
            }
        )
    return pd.DataFrame(rows)

def get_run_number_col(df: pd.DataFrame) -> str:
    """Dynamically find the run number column."""
    for col in POSSIBLE_RUN_COLS:
        if col in df.columns:
            return col
    st.warning("No run number column found (tried: TTM Number, Run Number, etc.). Using index as 'Run ID'.")
    return "Run ID"

def build_details(trips: pd.DataFrame, schedules: list, neighbors: dict) -> pd.DataFrame:
    rows = []
    run_col = get_run_number_col(trips)

    for s in schedules:
        sid = s["id"]
        idxs = s["trip_indices"]
        cum_km = 0.0

        for order, idx in enumerate(idxs, start=1):
            if run_col == "Run ID":
                run_number = idx  # Fallback to index
            else:
                run_number = trips.loc[idx, run_col]
            pickup_dt = trips.loc[idx, "pickup_dt"]
            drop_dt = trips.loc[idx, "drop_dt"]
            pick_zone = int(trips.loc[idx, "First Pickup Zone"])
            drop_zone = int(trips.loc[idx, "Last Dropoff Zone"])
            km = float(trips.loc[idx, "KM"])
            cum_km += km

            if order == 1:
                justification = "First trip in schedule (earliest uncovered trip)."
            else:
                prev_idx = idxs[order - 2]
                prev_drop_time = trips.loc[prev_idx, "drop_dt"]
                prev_drop_zone = int(trips.loc[prev_idx, "Last Dropoff Zone"])
                delta_min = int((pickup_dt - prev_drop_time).total_seconds() / 60.0)
                dist = zone_distance(neighbors, prev_drop_zone, pick_zone, max_depth=MAX_ZONE_DEPTH)

                if dist in GAP_RULES:
                    gap_rule = f"{GAP_RULES[dist]}-minute gap rule (distance {dist}-hop zone)"
                else:
                    gap_rule = "zone distance exceeded allowed range (this should not happen)"

                justification = (
                    f"Pickup {delta_min} mins after previous drop; "
                    f"pickup zone distance {dist} from dropoff zone {prev_drop_zone} "
                    f"({gap_rule})."
                )

            rows.append(
                {
                    "Schedule_ID": sid,
                    "Trip Order": order,
                    "Run Number": run_number,
                    "Pickup Time": pickup_dt.strftime("%H:%M"),
                    "Pick Zone": pick_zone,
                    "Dropoff Zone": drop_zone,
                    "Dropoff Time": drop_dt.strftime("%H:%M"),
                    "Trip KM": round(km, 3),
                    "Schedule Total KM": round(cum_km, 3),
                    "Linkage Justification": justification,
                }
            )

    return pd.DataFrame(rows)

# New functions for state management and updates

def load_existing_schedules(schedule_json_path: Optional[str] = None, uploaded_file=None) -> Optional[List[Dict]]:
    """Load prior schedules from JSON for incremental updates."""
    if uploaded_file:
        return json.load(uploaded_file)
    if not schedule_json_path:
        return None
    with open(schedule_json_path, 'r') as f:
        return json.load(f)

def apply_updates(trips: pd.DataFrame, updates: Optional[pd.DataFrame], existing_schedules: Optional[List[Dict]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process cancellations/add-ons: Filter trips, mark status."""
    updated_trips = trips.copy()
    updated_trips['Status'] = 'active'  # Default
    run_col = get_run_number_col(updated_trips)
    
    if updates is not None:
        updates_run_col = get_run_number_col(updates)  # Assume same col name in updates
        for _, row in updates.iterrows():
            run_num = row[updates_run_col]
            mask = updated_trips[run_col] == run_num
            if row['Status'] == 'canceled':
                updated_trips.loc[mask, 'Status'] = 'canceled'
            elif row['Status'] == 'added':
                # Append new row (assume updates has full trip cols for add-ons)
                new_row = row.to_dict()
                new_row['Status'] = 'active'
                updated_trips = pd.concat([updated_trips, pd.DataFrame([new_row])], ignore_index=True)
    
    # Filter to active only for scheduling
    active_trips = updated_trips[updated_trips['Status'] == 'active'].copy()
    active_trips = active_trips.sort_values("pickup_dt").reset_index(drop=True)
    
    # If existing schedules, remove canceled trips from them (for repair logic below)
    if existing_schedules:
        for sched in existing_schedules:
            sched['trip_indices'] = [i for i in sched['trip_indices'] if updated_trips.loc[i, 'Status'] != 'canceled']
    
    return active_trips, updated_trips  # Return active for build, full for details

def repair_schedules(active_trips: pd.DataFrame, neighbors: dict, existing_schedules: List[Dict], available_drivers: int = 0) -> List[Dict]:
    """Incremental rebuild: Repair existing, then add new. Cap new schedules by available_drivers."""
    all_schedules = existing_schedules.copy()  # Start with priors (sans cancels)
    unassigned = set(active_trips.index.tolist())
    
    # Mark already assigned (from repaired existing)
    for sched in all_schedules:
        for idx in sched['trip_indices']:
            if idx in unassigned:
                unassigned.remove(idx)
    
    # Filter out empties
    all_schedules = [s for s in all_schedules if len(s['trip_indices']) > 0]
    
    # Build new for unassigned, but limit to available_drivers
    new_schedules = build_schedules_from_unassigned(active_trips, neighbors, list(unassigned))
    all_schedules.extend(new_schedules[:available_drivers])  # Cap new ones
    
    return all_schedules

# AI Suggestion Hook (with OpenAI)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Graceful if not installed

def get_ai_suggestions(affected_schedules: List[Dict], available_drivers: int, api_key: Optional[str] = None) -> str:
    """Query OpenAI for reroute ideas if disruptions > threshold."""
    if not api_key or not OpenAI:
        return "OpenAI not configured (add key to secrets.toml and install openai)."
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are a logistics AI assistant specializing in driver scheduling. 
    Analyze these disrupted schedules: {json.dumps(affected_schedules[:3], indent=2)}.
    You have {available_drivers} additional drivers available.
    
    Suggest 2-3 practical fixes to minimize total KM, respect time gaps by zone distance (10 min for 0/1-hop, 15 for 2-hop, 20 for 3-hop, 25 for 4-hop), and limit zone hops to 4. 
    Prioritize swaps, merges, or new assignments for affected trips.
    Output as a concise bulleted list:
    - Suggestion 1: [Brief action] ([Estimated impact, e.g., saves 5km])
    - etc."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4o-mini" for better reasoning
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3  # Low for deterministic suggestions
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

# Sample Files Data (for downloads)
SAMPLE_TRIPS_CSV = """TTM Number,First Pickup Time,Last Dropoff Time,First Pickup Zone,Last Dropoff Zone,KM
001,08:00:00,09:00:00,1,2,25.5
002,09:20:00,10:30:00,2,3,30.0
003,10:45:00,11:45:00,3,1,20.0
"""
SAMPLE_UPDATES_CSV = """Run Number,Status,First Pickup Time,Last Dropoff Time,First Pickup Zone,Last Dropoff Zone,KM
001,canceled,,,,
999,added,11:00:00,12:00:00,4,5,20.0
"""
SAMPLE_JSON = """[
  {
    "id": "SCH-001",
    "trip_indices": [0, 1]
  },
  {
    "id": "SCH-002",
    "trip_indices": [2]
  }
]
"""

# Streamlit app
st.title("Live Driver Schedule Builder V2")

# Global zone cache
if "neighbors" not in st.session_state:
    st.session_state.neighbors = None

# Zone file uploader (global, cached across tabs)
st.header("Zone Graph (Upload Once)")
if st.session_state.neighbors is None:
    zones_file = st.file_uploader("Upload zones file (CSV or XLSX) - upload once and reuse", type=["csv", "xlsx"])
    if zones_file:
        with st.spinner("Loading zone graph..."):
            st.session_state.neighbors = load_zone_graph(zones_file)
        st.success("Zone graph loaded and cached!")
else:
    st.info("Zone graph already loaded from previous upload.")
    if st.button("Reload zone file"):
        st.session_state.neighbors = None
        st.rerun()

if st.session_state.neighbors is None:
    st.stop()  # Can't proceed without zones

# Main Tabs
tab1, tab2 = st.tabs(["Part 1: Initial Scheduling", "Part 2: Rescheduling"])

with tab1:
    st.header("Build Fresh Schedules (e.g., 6am Baseline)")
    
    # Sample Trips Download
    st.subheader("Sample Files")
    st.download_button(
        label="Download Sample Trips CSV",
        data=SAMPLE_TRIPS_CSV,
        file_name='sample_trips.csv',
        mime='text/csv'
    )
    
    # Trips uploader
    trips_file = st.file_uploader("Upload Today's Trips CSV", type="csv")
    
    if st.button("Build Schedules") and trips_file:
        with st.spinner("Building schedules..."):
            trips = load_trips(trips_file)
            schedules = build_schedules(trips, st.session_state.neighbors)
            summary_df = build_summary(trips, schedules)
            details_df = build_details(trips, schedules, st.session_state.neighbors)

        st.success(f"Generated {len(schedules)} schedules!")

        # Display summary table
        st.subheader("Schedule Summary")
        st.dataframe(summary_df)

        # Download buttons
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Summary CSV",
            data=csv_summary,
            file_name='schedule_summary.csv',
            mime='text/csv'
        )

        csv_details = details_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Details CSV",
            data=csv_details,
            file_name='full_schedule_details.csv',
            mime='text/csv'
        )

        # Optional: Display details table (truncated for view)
        with st.expander("View Details Table"):
            st.dataframe(details_df)
        
        # Download JSON for Part 2
        st.download_button(
            label="Download Schedules JSON (Save for Rescheduling)",
            data=json.dumps(schedules, indent=2).encode('utf-8'),
            file_name='current_schedules.json',
            mime='application/json'
        )

    else:
        if not trips_file:
            st.warning("Please upload the trips CSV.")

with tab2:
    st.header("Reschedule with Cancellations & Add-Ons (e.g., 11am Updates)")
    
    # Sample Files Downloads
    st.subheader("Sample Files")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Sample Trips CSV",
            data=SAMPLE_TRIPS_CSV,
            file_name='sample_trips.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="Sample Updates CSV (Cancellations/Add-Ons)",
            data=SAMPLE_UPDATES_CSV,
            file_name='sample_updates.csv',
            mime='text/csv'
        )
    with col3:
        st.download_button(
            label="Sample Schedules JSON",
            data=SAMPLE_JSON,
            file_name='sample_schedules.json',
            mime='application/json'
        )
    
    # Available drivers slider
    available_drivers = st.slider("Available Additional Drivers", 0, 10, 2)
    
    # Existing schedules loader
    existing_json = st.file_uploader("Upload Prior Schedules JSON (from Part 1)", type="json")
    existing_schedules = None
    if existing_json:
        existing_schedules = json.load(existing_json)
        st.success("Prior schedules loaded!")
    
    # Trips & Updates uploaders
    trips_file = st.file_uploader("Upload Trips CSV (for context/active trips)", type="csv")
    updates_file = st.file_uploader("Upload Updates CSV (Cancellations/Add-Ons)", type="csv")
    
    # Updates Format Info
    st.info("""
    **Updates CSV Format**:
    - `Run Number` (or matching col like TTM Number): ID from trips for cancels.
    - `Status`: "canceled" (leave other cols blank) or "added" (fill full trip details).
    Example row for cancel: Run Number=001, Status=canceled
    Example row for add-on: Run Number=999, Status=added, First Pickup Time=11:00:00, etc.
    """)
    
    if st.button("Rebuild Schedules") and trips_file:
        with st.spinner("Processing updates & rebuilding..."):
            trips = load_trips(trips_file)
            updates = pd.read_csv(updates_file) if updates_file else None
            existing = existing_schedules
            active_trips, full_trips = apply_updates(trips, updates, existing)
            
            schedules = repair_schedules(active_trips, st.session_state.neighbors, existing or [], available_drivers)
            
            # Check for disruptions (empty schedules)
            disrupted = [s for s in schedules if len(s['trip_indices']) == 0]
            if len(disrupted) > 0:  # Simplified threshold
                ai_sugs = get_ai_suggestions(disrupted, available_drivers, st.secrets.get("OPENAI_API_KEY", None))
                st.warning(f"Disruptions detected! AI Suggestions: {ai_sugs}")
            
            summary_df = build_summary(full_trips, schedules)
            details_df = build_details(full_trips, schedules, st.session_state.neighbors)

        st.success(f"Rebuilt {len(schedules)} schedules (using {available_drivers} extra drivers).")

        # Display summary table
        st.subheader("Updated Schedule Summary")
        st.dataframe(summary_df)

        # Download buttons
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Updated Summary CSV",
            data=csv_summary,
            file_name='updated_schedule_summary.csv',
            mime='text/csv'
        )

        csv_details = details_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Updated Details CSV",
            data=csv_details,
            file_name='updated_full_schedule_details.csv',
            mime='text/csv'
        )

        # Optional: Display details table (truncated for view)
        with st.expander("View Updated Details Table"):
            st.dataframe(details_df)
        
        # Download new JSON for next rebuild
        st.download_button(
            label="Download Updated Schedules JSON (for Next Reschedule)",
            data=json.dumps(schedules, indent=2).encode('utf-8'),
            file_name='updated_schedules.json',
            mime='application/json'
        )

    else:
        if not trips_file:
            st.warning("Please upload the trips CSV.")
        if not existing_json:
            st.warning("Upload prior JSON for full rescheduling (or run Part 1 first).")
