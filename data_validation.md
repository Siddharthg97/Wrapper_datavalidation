**isinstance(object, classinfo)**
  ,
object: Any value or variable you want to examine.
classinfo: A type (like int, str, list, etc.), a custom class, or even a tuple of types.
  
isinstance(5, int)                # True
isinstance("Hi", str)             # True
isinstance([1,2,3], (list, tuple))# True for either type


**Key Error**
    # Ensure date is datetime
    if date_column not in df_month.columns:
        raise KeyError(f"Date column '{date_column}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df_month[date_column]):
        df_month[date_column] = pd.to_datetime(df_month[date_column])
