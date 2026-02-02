import json
import re


def convert_percent_to_decimal(percent):
    try:
        return float(percent)
    except (ValueError, TypeError) as e:
        print(f"[convert_percent_to_decimal] error: {e}")
        return None


def parse_operations(response):
    if response == "DONE":
        return {"type": "DONE", "data": None}

    if response.startswith("CLICK"):
        match = re.search(r"CLICK \{ (.+) \}", response)
        if not match:
            return {"type": "UNKNOWN", "data": response}
        try:
            click_data_json = json.loads(f"{{{match.group(1)}}}")
        except json.JSONDecodeError:
            return {"type": "UNKNOWN", "data": response}
        return {"type": "CLICK", "data": click_data_json}

    if response.startswith("TYPE"):
        match = re.search(r"TYPE (.+)", response, re.DOTALL)
        if not match:
            match = re.search(r'TYPE "(.+)"', response, re.DOTALL)
        if not match:
            return {"type": "UNKNOWN", "data": response}
        return {"type": "TYPE", "data": match.group(1)}

    if response.startswith("SEARCH"):
        match = re.search(r'SEARCH "(.+)"', response)
        if not match:
            match = re.search(r"SEARCH (.+)", response)
        if not match:
            return {"type": "UNKNOWN", "data": response}
        return {"type": "SEARCH", "data": match.group(1)}

    return {"type": "UNKNOWN", "data": response}
