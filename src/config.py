"""Shared settings for the whole project (paths, seed, column lists)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hotels.csv"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = ROOT / "models"

for p in (RESULTS, FIGS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- Leakage guard -------------------------------------------------------
# Columns that are only known AFTER the booking outcome (cancel / check-in).
# They must never be used as features.
HARD_LEAK_COLS = ["reservation_status", "reservation_status_date"]
AFTER_BOOKING_COLS = ["assigned_room_type", "booking_changes"]

# ---- Feature lists -------------------------------------------------------
NUM_COLS = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
    "adults", "children", "babies", "total_guests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "required_car_parking_spaces",
    "total_of_special_requests",
]
BIN_COLS = ["is_repeated_guest", "has_agent", "has_company",
            "has_children", "is_weekend_arrival"]
CAT_COLS = ["hotel", "arrival_date_month", "arrival_weekday", "meal",
            "country", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "customer_type"]

# adr (price) is the regression target, but a normal known-at-booking feature
# for the cancellation (classification) problem.
CLF_NUM_COLS = NUM_COLS + ["adr"]
