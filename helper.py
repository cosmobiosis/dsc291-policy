
from datetime import datetime
from datetime import timezone

def convert_timestr_to_epoch(time_str):
    parsed = time_str.split(' ')
    dates = parsed[0].split("/")
    year = int(dates[2])
    month = int(dates[0])
    day = int(dates[1])

    clocks = parsed[1].split(":")
    hour = int(clocks[0])
    minute = int(clocks[1])
    second = int(clocks[2])
    if parsed[2] == "PM" and hour != 12:
        hour += 12
    
    dt = datetime(year, month, day, hour, minute, second)
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
    return int(timestamp)

def convert_timestr_to_hours(time_str):
    parsed = time_str.split(' ')
    clocks = parsed[1].split(":")
    hour = int(clocks[0])
    if parsed[2] == "PM":
        hour += 12
    return int(hour)

print(convert_timestr_to_epoch("05/03/2016 11:40:00 PM"))