import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime, timedelta
import re
import time # For potential delays

# --- Configuration ---
OUTPUT_CSV_FILE = 'economic_calendar_events.csv'
LOG_FILE = 'economic_calendar_scraper.log'
# Base URL, specific week/day might be appended
BASE_URL = 'https://www.forexfactory.com/'
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Connection': 'keep-alive',
    'DNT': '1' # Do Not Track
}
TARGET_CURRENCIES = ["USD", "EUR"] # Filter for these currencies

# Impact level mapping (based on typical ForexFactory CSS classes for impact icons)
IMPACT_MAP = {
    'high': 'High',    # Often class 'icon--ff-impact-red' or similar text in title
    'medium': 'Medium', # Often class 'icon--ff-impact-orange'
    'low': 'Low',      # Often class 'icon--ff-impact-yellow'
    'holiday': 'Holiday', # Often class 'icon--ff-impact-grey' or 'holiday' text
    'none': 'Non-Economic' # For other non-impact events if distinguishable
}
# Note: Actual class names will need to be verified by inspecting FF HTML.
# ForexFactory uses title attributes on impact icons, e.g., title="High Impact Expected"

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to console and file: {LOG_FILE}")
except Exception as e:
    logger.error(f"Failed to set up file logger for {LOG_FILE}: {e}", exc_info=True)
    logger.info("Logging to console only.")


def get_calendar_url_for_current_week():
    """
    Constructs the ForexFactory calendar URL for the current week.
    Example: https://www.forexfactory.com/calendar?week=nov26.2023
    """
    now = datetime.now()
    # FF week seems to start on Sunday.
    # current_sunday = now - timedelta(days=(now.weekday() + 1) % 7)
    # However, their week notation in URL is like "oct1.2023" which might be first day of that specific displayed week.
    # For simplicity, let's try to get the current day's calendar, which redirects to the week view.
    # Or, use the 'calendar.php' which seems to show current week by default.
    return BASE_URL + "calendar" # Simpler, usually defaults to current week.

def fetch_calendar_html(url):
    logger.info(f"Fetching calendar HTML from: {url}")
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        response.raise_for_status()
        logger.info(f"Successfully fetched HTML. Status: {response.status_code}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching HTML from {url}: {e}", exc_info=True)
        return None

def parse_date_from_ff(date_str, time_str, year):
    """
    Parses ForexFactory's date representation (e.g., "Mon", "Nov 27", "Today") into YYYY-MM-DD.
    This is tricky because the date is relative to the week being viewed.
    The current approach will be simplified: it will rely on the date provided in the row if full,
    otherwise it's more complex and needs context of previous rows.
    For now, this will be a placeholder for more robust parsing.
    A simpler strategy: FF calendar table rows often have a "date" cell that repeats.
    """
    # This helper is largely superseded by parsing the full date headers directly in parse_calendar_events.
    # Kept for reference or future use if individual date cell parsing becomes necessary again.

    if not date_str or date_str.lower() == "today" or date_str.lower() == "tomorrow":
        return None

    try:
        dt_obj = datetime.strptime(f"{date_str} {year}", "%b %d %Y")
        return dt_obj.strftime("%Y-%m-%d")
    except ValueError:
        logger.debug(f"Could not parse date string '{date_str}' with year '{year}' directly.")
        return None


def parse_calendar_events(html_content): # current_year removed, as full date is parsed from header
    """
    Parses the HTML content of the ForexFactory calendar page to extract events.
    """
    if not html_content:
        logger.error("HTML content is empty, cannot parse.")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    events = []

    # ForexFactory calendar is typically a table with class 'calendar__table'
    # Rows are 'calendar__row'. Date rows are different from event rows.
    # This requires careful inspection of the live ForexFactory HTML structure.
    # Example structure (might change):
    # <tr class="calendar__row calendar__date-row"><td class="calendar__date" colspan="7"><span>Monday, November 27, 2023</span></td></tr>
    # <tr class="calendar__row calendar_row--grey calendar_row--holiday" data-eventid="123">...</tr>
    # <tr class="calendar__row" data-eventid="124">
    #   <td class="calendar__cell calendar__date date">Nov 27</td>  <-- This may NOT be present or may be just day like "Mon"
    #   <td class="calendar__cell calendar__time time">5:00am</td>
    #   <td class="calendar__cell calendar__currency currency">JPY</td>
    #   <td class="calendar__cell calendar__impact impact"><span title="Low Impact Expected" class="icon icon--ff-impact-yel"></span></td>
    #   <td class="calendar__cell calendar__event event"><div><span>Event Name</span></div></td>  <-- Event name can be nested
    #   ... details (actual, forecast, previous) ...
    # </tr>

    calendar_table = soup.find('table', class_='calendar__table')
    if not calendar_table:
        logger.warning("Could not find the main calendar table (class 'calendar__table'). Page structure might have changed or page not loaded correctly.")
        return []

    current_processing_date_str = None # Stores YYYY-MM-DD string

    for row in calendar_table.find_all('tr', class_=re.compile(r'calendar__row')):
        try:
            row_classes = row.get('class', [])

            # Date Header Row Processing
            if 'calendar__date-row' in row_classes:
                # FF has two types of date rows: one with full "Monday, November 27, 2023"
                # and another simpler one for a new day within the same week view.
                # Let's try to find the most specific date element.
                date_span = row.find('span', class_='date') # Often the full date is in a span
                if not date_span: # Sometimes it's directly in the td
                    date_td = row.find('td', class_='calendar__date')
                    if date_td : date_span = date_td

                if date_span:
                    date_full_str = date_span.text.strip()
                    try:
                        # Example: "Monday, November 27, 2023" or "November 27, 2023"
                        # Try multiple formats if necessary
                        parsed_dt = None
                        for fmt in ("%A, %B %d, %Y", "%B %d, %Y"): # Day of week optional
                            try:
                                parsed_dt = datetime.strptime(date_full_str, fmt)
                                break
                            except ValueError:
                                continue
                        if parsed_dt:
                            current_processing_date_str = parsed_dt.strftime("%Y-%m-%d")
                            logger.debug(f"Date header processed. Current date set to: {current_processing_date_str}")
                        else:
                             logger.warning(f"Could not parse date header string: '{date_full_str}'")
                    except Exception as e_date_parse:
                        logger.warning(f"Error parsing date header '{date_full_str}': {e_date_parse}")
                else:
                    logger.debug("Found a date row, but no 'span.date' or 'td.calendar__date' to parse.")
                continue # This row is a date header, not an event

            # Event Row Processing
            event_id = row.get('data-eventid')
            if not event_id: # Skip if not an event row (e.g. spacer rows, etc.)
                continue

            if not current_processing_date_str:
                logger.warning(f"Skipping event row (ID: {event_id}) because no current date has been established from headers.")
                continue

            event_date = current_processing_date_str # Date for this event is from the last header

            time_cell = row.find('td', class_='time')
            event_time_str = time_cell.text.strip() if time_cell else "All Day"
            event_time_final = "All Day" # Default

            if "all day" not in event_time_str.lower() and event_time_str:
                try:
                    dt_time_obj = datetime.strptime(event_time_str.upper(), "%I:%M%p") # Handles am/pm, case-insensitive
                    event_time_final = dt_time_obj.strftime("%H:%M")
                except ValueError:
                    try:
                        dt_time_obj = datetime.strptime(event_time_str, "%H:%M") # Handles 24-hour format
                        event_time_final = dt_time_obj.strftime("%H:%M")
                    except ValueError:
                        logger.debug(f"Could not parse time '{event_time_str}' for event ID {event_id} on {event_date}. Defaulting to 'All Day'.")

            currency_cell = row.find('td', class_='currency')
            currency = currency_cell.text.strip() if currency_cell else ""

            if currency not in TARGET_CURRENCIES:
                continue

            impact_str = "Non-Economic" # Default
            impact_cell = row.find('td', class_='impact')
            if impact_cell:
                impact_span = impact_cell.find('span', title=True) # Find span with a title attribute
                if impact_span:
                    title_text = impact_span['title'].lower()
                    if 'high impact' in title_text: impact_str = IMPACT_MAP['high']
                    elif 'medium impact' in title_text: impact_str = IMPACT_MAP['medium']
                    elif 'low impact' in title_text: impact_str = IMPACT_MAP['low']
                    # Holiday check by title might be less reliable than class on row

            # Check for holiday class on the row itself, which is more reliable for holidays
            if 'calendar__row--holiday' in row_classes:
                impact_str = IMPACT_MAP['holiday']

            event_name_cell = row.find('td', class_='event')
            # Event name might be in a span, or a nested span, or directly in the td.
            # Try to find the most specific element containing the text.
            event_name_span = event_name_cell.find('span') if event_name_cell else None
            if event_name_span and event_name_span.find('span'): # If there's a span inside a span
                event_name = event_name_span.find('span').text.strip()
            elif event_name_span: # If there's just one span
                event_name = event_name_span.text.strip()
            elif event_name_cell: # If text is directly in td
                event_name = event_name_cell.text.strip()
            else:
                event_name = "N/A"

            forecast_cell = row.find('td', class_='forecast')
            forecast_val = forecast_cell.text.strip() if forecast_cell else ""

            previous_cell = row.find('td', class_='previous')
            previous_val = previous_cell.text.strip() if previous_cell else ""

            event_data = {
                'date': event_date, # YYYY-MM-DD
                'time_et': event_time_final, # HH:MM or "All Day"
                'currency': currency,
                'impact': impact_str,
                'event_name': event_name,
                'forecast': forecast_val,
                'previous': previous_val
            }
            events.append(event_data)
            logger.debug(f"Parsed event: {event_data}")

        except Exception as e:
            logger.error(f"Error parsing an event row (ID: {event_id if event_id else 'N/A'}): {e}. Row HTML (partial): {str(row)[:300]}", exc_info=True)
            continue

    logger.info(f"Total events parsed (pre-filter by currency): {len(events)} (Note: currency filter applied during parsing).")
    return events


def main():
    logger.info("--- Starting ForexFactory Economic Calendar Scraper ---")

    calendar_url = get_calendar_url_for_current_week()
    html_content = fetch_calendar_html(calendar_url)

    if not html_content:
        logger.error("Failed to fetch HTML content. Exiting.")
        return

    # current_year = datetime.now().year # No longer needed as full date is parsed from header
    parsed_events = parse_calendar_events(html_content)

    if parsed_events:
        df = pd.DataFrame(parsed_events)
        # Ensure desired column order
        df = df[['date', 'time_et', 'currency', 'impact', 'event_name', 'forecast', 'previous']]
        try:
            df.to_csv(OUTPUT_CSV_FILE, index=False)
            logger.info(f"Successfully saved {len(df)} events to {OUTPUT_CSV_FILE}")
            logger.info("Reminder: Event times are based on ForexFactory display, assumed to be Eastern Time (ET). No UTC conversion performed by this script.")
        except Exception as e:
            logger.error(f"Error saving data to CSV {OUTPUT_CSV_FILE}: {e}", exc_info=True)
    else:
        logger.info("No events were parsed or no events matched the target criteria.")

    logger.info("--- Economic Calendar Scraper Finished ---")

if __name__ == '__main__':
    # Consider adding a small delay before starting to be polite to the server, though for a manual run it's minor.
    # time.sleep(1)
    main()
