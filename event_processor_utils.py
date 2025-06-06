import pandas as pd
import pytz # For timezone handling
from datetime import datetime, date, time
import logging
import os

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Note: Logger configuration (level, handlers) is typically done in the main application
# that uses this utility. For standalone testing, basicConfig can be used here.
# For now, assume calling script configures logger. If run standalone, no output unless configured.

# --- Constants ---
DEFAULT_CALENDAR_CSV = 'economic_calendar_events.csv'
EASTERN_TIMEZONE = 'America/New_York'

def load_calendar_events(csv_filepath=DEFAULT_CALENDAR_CSV):
    """
    Loads economic calendar events from a CSV file.

    Args:
        csv_filepath (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing calendar events, or None if loading fails.
    """
    logger.info(f"Attempting to load economic calendar events from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        logger.error(f"Calendar events file not found: {csv_filepath}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
        logger.info(f"Successfully loaded {len(df)} events from {csv_filepath}.")

        # Basic validation for required columns
        required_cols = ['date', 'time_et', 'currency', 'impact', 'event_name']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing one or more required columns ({required_cols}) in {csv_filepath}.")
            return None

        # Ensure 'date' is string to be combined later, and 'time_et' is string
        df['date'] = df['date'].astype(str)
        df['time_et'] = df['time_et'].astype(str)

        return df
    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        logger.error(f"Calendar events file not found (FileNotFoundError): {csv_filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Calendar events file is empty: {csv_filepath}")
        return pd.DataFrame(columns=required_cols) # Return empty df with expected columns
    except Exception as e:
        logger.error(f"Error loading calendar events from {csv_filepath}: {e}", exc_info=True)
        return None

def get_relevant_events_for_day(events_df, target_date_utc: date,
                                target_currencies=['USD', 'EUR'],
                                target_impacts=['High']):
    """
    Filters calendar events for a specific UTC date, currencies, and impact levels,
    after converting their times from ET to UTC.

    Args:
        events_df (pd.DataFrame): DataFrame loaded by load_calendar_events.
        target_date_utc (datetime.date): The UTC date to filter events for.
        target_currencies (list): List of currency strings to filter for (e.g., ['USD', 'EUR']).
        target_impacts (list): List of impact level strings to filter for (e.g., ['High']).

    Returns:
        pandas.DataFrame: Filtered DataFrame with events for the target UTC date,
                          including a 'datetime_utc' column. Returns empty DataFrame if no relevant events.
    """
    if events_df is None or events_df.empty:
        logger.info("Events DataFrame is empty or None. No relevant events to process.")
        return pd.DataFrame()

    # Make a copy to avoid modifying the original DataFrame passed to the function
    df = events_df.copy()

    logger.debug(f"Processing {len(df)} events for target UTC date: {target_date_utc.isoformat()}")

    # 1. Handle "All Day" events in 'time_et' - filter them out as they don't have a specific time
    df = df[df['time_et'].str.lower() != 'all day']
    if df.empty:
        logger.debug("No events remaining after filtering out 'All Day' events.")
        return pd.DataFrame()
    logger.debug(f"{len(df)} events remaining after filtering 'All Day'.")

    # 2. Combine 'date' and 'time_et' and parse to datetime objects localized to Eastern Time
    # Assuming 'date' is YYYY-MM-DD and 'time_et' is HH:MM
    df['datetime_et_str'] = df['date'] + ' ' + df['time_et']

    # Convert to datetime, attempting to parse. Errors will result in NaT.
    # Using errors='coerce' will turn unparseable formats into NaT
    df['datetime_et_aware'] = pd.to_datetime(df['datetime_et_str'], errors='coerce')

    # Drop rows where datetime parsing failed
    df.dropna(subset=['datetime_et_aware'], inplace=True)
    if df.empty:
        logger.debug("No events remaining after failing to parse some date/time strings.")
        return pd.DataFrame()
    logger.debug(f"{len(df)} events remaining after initial datetime parsing.")

    # Localize to Eastern Time, handling DST ambiguities
    # 'ambiguous': pass False to raise exception, True to infer based on DST, or 'NaT' to assign NaT
    # 'nonexistent': times that don't exist during DST spring-forward
    try:
        et_tz = pytz.timezone(EASTERN_TIMEZONE)
        df['datetime_et_aware'] = df['datetime_et_aware'].apply(lambda dt: et_tz.localize(dt, is_dst=None)) # is_dst=None handles ambiguity better for many cases
    except pytz.exceptions.AmbiguousTimeError:
        logger.warning(f"Encountered ambiguous times during ET localization (DST fall-back). These may be handled by pandas or pytz default, or cause issues.")
        # Pandas .dt.tz_localize can also handle this:
        # df['datetime_et_aware'] = df['datetime_et_aware'].dt.tz_localize(EASTERN_TIMEZONE, ambiguous='infer', nonexistent='NaT')
        # For now, simple apply. If this becomes an issue, more robust handling for ambiguous times needed.
        # A common way is to use 'infer' or a specific offset during ambiguous periods if known.
        # For now, if et_tz.localize fails due to ambiguity, it will raise and be caught by the broader except block.
        # Let's refine this to use pandas' built-in capability which is often more robust:
    df['datetime_et_aware'] = pd.to_datetime(df['datetime_et_str'], errors='coerce').dt.tz_localize(EASTERN_TIMEZONE, ambiguous='infer', nonexistent='NaT')
    df.dropna(subset=['datetime_et_aware'], inplace=True) # Drop if NaT from ambiguous/nonexistent
    if df.empty:
        logger.debug("No events remaining after ET localization (possibly due to ambiguous/nonexistent times).")
        return pd.DataFrame()
    logger.debug(f"{len(df)} events remaining after ET localization.")


    # 3. Convert to UTC
    df['datetime_utc'] = df['datetime_et_aware'].dt.tz_convert('UTC')
    logger.debug(f"Converted event times to UTC. Sample UTC timestamp: {df['datetime_utc'].iloc[0] if not df.empty else 'N/A'}")

    # 4. Filter by target_date_utc
    # Ensure target_date_utc is a date object
    if not isinstance(target_date_utc, date):
        logger.error(f"target_date_utc must be a datetime.date object, got {type(target_date_utc)}")
        return pd.DataFrame()

    df_filtered = df[df['datetime_utc'].dt.date == target_date_utc].copy() # Use .copy() to avoid SettingWithCopyWarning
    logger.debug(f"{len(df_filtered)} events found for target UTC date: {target_date_utc.isoformat()}")

    # 5. Filter by target_currencies
    if target_currencies:
        df_filtered = df_filtered[df_filtered['currency'].isin(target_currencies)]
        logger.debug(f"{len(df_filtered)} events remaining after currency filter ({target_currencies}).")

    # 6. Filter by target_impacts
    if target_impacts:
        df_filtered = df_filtered[df_filtered['impact'].isin(target_impacts)]
        logger.debug(f"{len(df_filtered)} events remaining after impact filter ({target_impacts}).")

    if not df_filtered.empty:
        logger.info(f"Found {len(df_filtered)} relevant events for {target_date_utc.isoformat()} with currencies {target_currencies} and impacts {target_impacts}.")
    else:
        logger.info(f"No relevant events found for {target_date_utc.isoformat()} with specified criteria.")

    return df_filtered


def is_event_blackout_active(processed_events_df_utc_today):
    """
    Checks if there are any high-impact events scheduled for the current UTC day.

    Args:
        processed_events_df_utc_today (pd.DataFrame): DataFrame of relevant events for the current UTC day,
                                                     as returned by get_relevant_events_for_day.

    Returns:
        bool: True if there's at least one event indicating a blackout, False otherwise.
    """
    if processed_events_df_utc_today is None:
        logger.debug("is_event_blackout_active received None DataFrame, assuming no blackout.")
        return False

    if not processed_events_df_utc_today.empty:
        logger.info(f"Blackout ACTIVE: {len(processed_events_df_utc_today)} high-impact event(s) found for the current UTC day.")
        # Log details of the first few events causing blackout for better traceability
        for index, row in processed_events_df_utc_today.head(3).iterrows():
            logger.info(f"  - Event: {row['event_name']} ({row['currency']}) at {row['datetime_utc'].strftime('%H:%M')} UTC")
        return True
    else:
        logger.info("Blackout INACTIVE: No high-impact events found for the current UTC day with specified criteria.")
        return False

if __name__ == '__main__':
    # --- Basic Test for the utility module ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("--- Testing event_processor_utils.py ---")

    # Create a dummy economic_calendar_events.csv for testing
    dummy_data = {
        'date': ['2023-11-20', '2023-11-20', '2023-11-20', '2023-11-21', '2023-11-21', '2023-11-21', '2023-11-21'],
        'time_et': ['08:30', '10:00', '14:00', '09:00', 'All Day', '11:00', '15:30'],
        'currency': ['USD', 'EUR', 'USD', 'GBP', 'USD', 'EUR', 'USD'],
        'impact': ['High', 'Medium', 'High', 'Low', 'Holiday', 'High', 'Low'],
        'event_name': ['US CPI', 'Eurozone ZEW', 'FOMC Minutes', 'UK Retail Sales', 'US Thanksgiving', 'German Ifo', 'US ISM Non-Mfg'],
        'forecast': ['0.3%', '10.0', '', '0.5%', '', '90.5', '55.0'],
        'previous': ['0.4%', '9.5', '', '0.2%', '', '90.0', '54.5']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'dummy_test_calendar.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)
    logger.info(f"Created dummy calendar file: {dummy_csv_path}")

    # Test load_calendar_events
    loaded_df = load_calendar_events(csv_filepath=dummy_csv_path)
    if loaded_df is not None:
        logger.info(f"Loaded dummy data. Shape: {loaded_df.shape}")
        # print(loaded_df.head()) # Using print for quick debug in main, logger for functions

        # Test get_relevant_events_for_day
        # Assuming current UTC date is 2023-11-20 for testing purposes.
        # In real use, this would be datetime.now(timezone.utc).date()
        # ET for 2023-11-20 08:30 is 2023-11-20 13:30 UTC (assuming standard time, no DST change on this day)
        test_target_utc_date = date(2023, 11, 20)
        logger.info(f"\nTesting get_relevant_events_for_day for UTC date: {test_target_utc_date.isoformat()}")

        relevant_events_today = get_relevant_events_for_day(
            loaded_df,
            target_date_utc=test_target_utc_date,
            target_currencies=['USD', 'EUR'],
            target_impacts=['High', 'Medium'] # Include Medium for more test data
        )

        if not relevant_events_today.empty:
            logger.info(f"Relevant events found for {test_target_utc_date.isoformat()}:\n{relevant_events_today[['datetime_utc', 'currency', 'impact', 'event_name']].to_string()}")

            # Test is_event_blackout_active (assuming default target_impacts=['High'] for this test)
            # Re-filter for just High impact for the blackout test logic
            high_impact_events_today = get_relevant_events_for_day(
                loaded_df,
                target_date_utc=test_target_utc_date,
                target_currencies=['USD', 'EUR'],
                target_impacts=['High']
            )
            blackout = is_event_blackout_active(high_impact_events_today)
            logger.info(f"Blackout active for {test_target_utc_date.isoformat()} (High impact USD/EUR)? {blackout}")

            # Test for a day with no high-impact USD/EUR events in dummy data
            test_target_utc_date_no_event = date(2023, 11, 21)
            # Note: 2023-11-21 has a "Holiday" for USD, and a "High" for EUR.
            # The 'Holiday' impact type is not in default target_impacts=['High'] for is_event_blackout_active
            # So, for this date, only the EUR High event should trigger blackout if EUR is targeted.
            logger.info(f"\nTesting get_relevant_events_for_day for UTC date: {test_target_utc_date_no_event.isoformat()}")
            high_impact_events_no_event_day = get_relevant_events_for_day(
                loaded_df,
                target_date_utc=test_target_utc_date_no_event,
                target_currencies=['USD'], # Test with only USD to see if holiday is filtered by default
                target_impacts=['High']
            )
            blackout_no_event = is_event_blackout_active(high_impact_events_no_event_day)
            logger.info(f"Blackout active for {test_target_utc_date_no_event.isoformat()} (High impact USD only)? {blackout_no_event}")

            eur_high_impact_events_day2 = get_relevant_events_for_day(
                loaded_df,
                target_date_utc=test_target_utc_date_no_event,
                target_currencies=['EUR'],
                target_impacts=['High']
            )
            blackout_eur_day2 = is_event_blackout_active(eur_high_impact_events_day2)
            logger.info(f"Blackout active for {test_target_utc_date_no_event.isoformat()} (High impact EUR only)? {blackout_eur_day2}")


    else:
        logger.error("Failed to load dummy calendar data for testing.")

    # Clean up dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
        logger.info(f"Removed dummy calendar file: {dummy_csv_path}")
