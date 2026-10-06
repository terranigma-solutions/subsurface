MAX_LE_JSON_DEPTH = 128


def validate_le_json_nesting(header_bytes):
    """Bound JSON container nesting independently of the interpreter's decoder."""
    depth = 0
    quoted = False
    escaped = False
    for character in header_bytes:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
        elif character == 34:
            quoted = True
        elif character in (91, 123):
            depth += 1
            if depth > MAX_LE_JSON_DEPTH:
                raise ValueError("Invalid LE JSON header: JSON header exceeds supported nesting depth")
        elif character in (93, 125):
            depth -= 1
