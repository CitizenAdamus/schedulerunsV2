# schedulerunsV2
Updated version of the schedule runs app 
# Live Driver Schedule Builder V2

An evolved Streamlit web app for dynamic driver scheduling, building on V1 with mid-day updates for cancellations and add-ons. Use the original V1 app for baseline runs; switch here when ready for live tweaks.

## Key V2 Upgrades
- **Incremental Rebuilds**: Load prior schedules (JSON), apply cancels/add-ons via updates CSV, and repair chains without full resets.
- **Additional Drivers**: Slider to cap new schedules based on extras available.
- **AI Suggestions**: Stub for Grok/OpenAI hooks on high-disruption events (e.g., "Suggest swaps for 3+ broken schedules").
- **State Persistence**: Export/import JSON for seamless 6am-to-11am handoffs.

## Setup & Run
1. `pip install -r requirements.txt`
2. `streamlit run schedule_builder_v2.py`
3. Upload zones (cached), trips, updates, and optional prior JSON.

## Usage Flow
- **Morning Baseline**: Upload full trips CSV → Build → Download JSON alongside CSVs.
- **Mid-Day Tweaks**: Upload same trips + updates CSV (cancels/add-ons) + prior JSON → Rebuild → Review AI flags → Export new JSON/CSVs.
- **Updates CSV Format**:
  | Run Number | Status  | First Pickup Time | Last Dropoff Time | First Pickup Zone | Last Dropoff Zone | KM |
  |------------|---------|-------------------|-------------------|-------------------|-------------------|----|
  | 001       | canceled|                   |                   |                   |                   |    |
  | 999       | added  | 11:00:00         | 12:00:00         | 4                 | 5                 | 20.0 |

## Algorithm Notes
- Repairs: Removes cancels from existing schedules, rebuilds gaps with unassigned/add-ons.
- Constraints: Same as V1 (125km, 12h, 2-hop zones, 10/15min gaps).
- AI Stub: Edit `get_ai_suggestions()` for real API calls.

## Deployment
- Streamlit Cloud: Link this repo for instant hosting.
- Prod Tip: Use Streamlit's secrets for API keys; integrate with n8n for driver pings.

## From V1?
V1 lives at [original-repo-link]. This is a parallel track—migrate when V2's battle-tested.

## Contributing
PRs welcome! Test with sample data in `/examples/` (add your own).

License: MIT
