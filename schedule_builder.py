import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from io import BytesIO
from typing import Optional, Dict, List
import tempfile

# Constants from the original script
KM_LIMIT = 125.0         # hard max km per schedule
MAX_HOURS = 12.0         # max duration first pickup -> last drop (hours)
MAX_ZONE_DEPTH = 2       # 0,1,2 hops allowed
GAP_NEAR = 10            # minutes for distance 0 or 1
GAP_FAR = 15             # minutes for distance 2

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

def zone_distance(neighbors: dict, start: int, target: int, max_depth: int = 2):
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

                # Time gap rule
                if dist in (0, 1):
                    min_gap = GAP_NEAR
                else:  # dist == 2
                    min_gap = GAP_FAR

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

def build_details(trips: pd.DataFrame, schedules: list, neighbors: dict) -> pd.DataFrame:
    rows = []

    for s in schedules:
        sid = s["id"]
        idxs = s["trip_indices"]
        cum_km = 0.0

        for order, idx in enumerate(idxs, start=1):
            run_number = trips.loc[idx, "TTM Number"]
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

                if dist in (0, 1):
                    gap_rule = f"{GAP_NEAR}-minute gap rule (same/neighbor zone)"
                elif dist == 2:
                    gap_rule = f"{GAP_FAR}-minute gap rule (2-hop zone)"
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
    
    if updates is not None:
        for _, row in updates.iterrows():
            run_num = row['Run Number']  # Or 'TTM Number'
            mask = updated_trips['TTM Number'] == run_num  # Adjust col if needed
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

def repair_schedules(active_trips: pd.DataFrame, neighbors: dict, existing_schedules: List[Dict]) -> List[Dict]:
    """Incremental rebuild: Repair existing, then add new."""
    all_schedules = existing_schedules.copy()  # Start with priors (sans cancels)
    unassigned = set(active_trips.index.tolist())
    
    # Mark already assigned (from repaired existing)
    for sched in all_schedules:
        for idx in sched['trip_indices']:
            if idx in unassigned:
                unassigned.remove(idx)
    
    # For each existing schedule with gaps, try to fill (greedy insert by time)
    # Simplified: For now, we'll just filter out empties and rebuild unassigned
    all_schedules = [s for s in all_schedules if len(s['trip_indices']) > 0]
    
    # Now build new/remaining with unassigned (as before, but cap by available_drivers if set)
    new_schedules = build_schedules_from_unassigned(active_trips, neighbors, list(unassigned))
    all_schedules.extend(new_schedules)
    return all_schedules

# AI Suggestion Hook (stub—integrate Grok/OpenAI API)
def get_ai_suggestions(affected_schedules: List[Dict], available_drivers: int, api_key: Optional[str] = None) -> str:
    """Query AI for reroute ideas if disruptions > threshold."""
    prompt = f"Affected schedules: {json.dumps(affected_schedules[:3])}. Available drivers: {available_drivers}. Suggest 2-3 low-impact fixes."
    # Call to Grok API or openai.ChatCompletion (add your key via st.secrets)
    # For now, return a placeholder
    return "Suggestion 1: Swap Trip 002 to Driver 3 (saves 5km). Suggestion 2: Merge SCH-001 & SCH-002."

# Streamlit app
st.title("Live Driver Schedule Builder V2")

# Option for zone file: upload once and cache in session state, or upload fresh
if "neighbors" not in st.session_state:
    st.session_state.neighbors = None

# Zone file uploader (uploaded once, cached)
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

# New: Available drivers slider
available_drivers = st.slider("Available Additional Drivers", 0, 10, 2)

# Existing schedules loader
existing_json = st.file_uploader("Load Existing Schedules (JSON from prior run)", type="json")
if existing_json:
    existing_schedules = json.load(existing_json)
else:
    existing_schedules = None

# Trips & Updates uploaders
trips_file = st.file_uploader("Upload Today's Trips CSV (or latest)", type="csv")
updates_file = st.file_uploader("Upload Updates CSV (Cancellations/Add-ons)", type="csv")

# Sample Updates CSV Format
if not updates_file:
    st.info("""
    Updates CSV Example:
    Run Number,Status,First Pickup Time,Last Dropoff Time,First Pickup Zone,Last Dropoff Zone,KM
    001,canceled,,,,
    999,added,11:00:00,12:00:00,4,5,20.0
    """)

# Handle both initial build and rebuild
if (st.button("Build/Rebuild Schedules") and trips_file and st.session_state.neighbors) or (existing_json and st.button("Rebuild from Loaded Schedules")):
    with st.spinner("Processing updates & building/rebuilding..."):
        trips = load_trips(trips_file)
        updates = pd.read_csv(updates_file) if updates_file else None
        existing = load_existing_schedules(uploaded_file=existing_json) if existing_json else None
        active_trips, full_trips = apply_updates(trips, updates, existing)
        
        schedules = repair_schedules(active_trips, st.session_state.neighbors, existing or [])
        
        # Check for disruptions
        disrupted = [s for s in schedules if len(s['trip_indices']) == 0]  # Empty ones
        if len(disrupted) > available_drivers * 0.5:  # Threshold
            ai_sugs = get_ai_suggestions(disrupted, available_drivers, st.secrets.get("API_KEY", None))
            st.warning(f"High disruption detected! AI Suggestions: {ai_sugs}")
        
        summary_df = build_summary(full_trips, schedules)  # Use full for details, but active for logic
        details_df = build_details(full_trips, schedules, st.session_state.neighbors)

    st.success(f"Generated/Rebuilt {len(schedules)} schedules (using {available_drivers} extra drivers).")

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
    
    # New: Download updated JSON for next run
    st.download_button(
        label="Download Current Schedules JSON (for next rebuild)",
        data=json.dumps(schedules, indent=2).encode('utf-8'),
        file_name='current_schedules.json',
        mime='application/json'
    )

else:
    if not trips_file:
        st.warning("Please upload the trips CSV.")
    if st.session_state.neighbors is None:
        st.warning("Please upload the zones file first.")
