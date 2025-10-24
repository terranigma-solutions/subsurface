import pandas as pd


def safe_convert_to_float32(df: pd.DataFrame, error_handling: str = 'raise') -> pd.DataFrame:
    """
    Convert DataFrame columns to float32, handling non-convertible columns.

    Args:
        df: DataFrame to convert
        error_handling: How to handle non-convertible columns:
            - 'raise': Raise an error
            - 'skip': Skip non-convertible columns
            - 'drop': Drop non-convertible columns

    Returns:
        DataFrame with converted columns
    """
    convertible, non_convertible = _check_convertible_to_float32(df)

    if non_convertible:
        if error_handling == 'raise':
            raise ValueError(
                f"Cannot convert columns to float32: {non_convertible}. "
                f"These columns contain non-numeric data."
            )
        elif error_handling == 'skip':
            # Only convert the convertible columns
            result = df.copy()
            for col in convertible:
                result[col] = df[col].astype('float32')
            return result
        elif error_handling == 'drop':
            # Drop non-convertible columns
            return df[convertible].astype('float32')
        else:
            raise ValueError(f"Invalid error_handling: {error_handling}")

    return df.astype('float32')


def _check_convertible_to_float32(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Check which columns in a DataFrame can be safely converted to float32.

    Returns:
        tuple: (convertible_columns, non_convertible_columns)
    """
    convertible = []
    non_convertible = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Already numeric, can convert
            convertible.append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            # Boolean can be converted (True->1.0, False->0.0)
            convertible.append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                convertible.append(col)
            except (ValueError, TypeError):
                non_convertible.append(col)
        else:
            # Other types (datetime, timedelta, categorical, etc.)
            non_convertible.append(col)

    return convertible, non_convertible
