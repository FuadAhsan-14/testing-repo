from datetime import datetime, timezone

def timestamp_now():
    return int(datetime.now(timezone.utc).timestamp())

def convertValue(value):
    valueType = type(value)
    if valueType == float:
        return int(value)
    return value

def serialize_fields(fields):
    result = []
    for key in fields:
        result.append({
            "name": key,
            "value": convertValue(fields[key]),
        })
    return result
