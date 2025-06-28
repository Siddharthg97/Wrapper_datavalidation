def champion_challenger_multivariate(
    multivariate_model_data: pd.DataFrame,
    multivariate_model_list: list,
    key: str,
    date_col: str,
    target_col: str
) -> pd.DataFrame:
    """
    Select the best performing multivariate model per key based on MAPE.
 
    Parameters:
    - data: DataFrame with actual values (must contain columns: key, date, target).
    - multivariate_model_data: DataFrame with multivariate forecasts.
    - multivariate_model_list: List of model names (e.g., ['lgbm', 'lgbm2']).
    - key: Grouping key column name (e.g., 'key').
    - date_col: Date column name (e.g., 'date').
    - target_col: Actuals column name (e.g., 'target').
 
    Returns:
    - DataFrame with MAPE, bias, and best model for each key.
    """
    # Rename target col to 'actual' for consistency
    # df_actuals = data[[key, date_col, target_col]].rename(columns={target_col: 'target'})
 
    # Join forecasts with actuals
    final_data = multivariate_model_data
 
    # Compute MAPE and bias
    mape_df = calculate_mape(final_data, multivariate_model_list, key=key)
 
    # Determine the best model (lowest MAPE)
    mape_df['min_error'] = mape_df[multivariate_model_list].min(axis=1)
 
    def get_best_model(row):
        for model in multivariate_model_list:
            if row[model] == row['min_error']:
                return model
        return None
 
    mape_df['best_model'] = mape_df.apply(get_best_model, axis=1)
 
    return mape_df

------------------------------------ metric calculation -------------------------------------------------------
def calculate_mape(final_data: pd.DataFrame, model_list: list, key: str) -> pd.DataFrame:
    """
    Calculate MAPE and mean bias for each model in a pandas DataFrame, grouped by the specified key.

    Parameters:
    -----------
    final_data : pd.DataFrame
        DataFrame with actuals and model predictions. Prediction columns must be in format '<model>_pred'.

    model_list : list
        List of model name strings (e.g., ['lgbm', 'lgbm2']).

    key : str
        Column name to group by (e.g., 'key').

    Returns:
    --------
    pd.DataFrame
        DataFrame with MAPE and mean bias for each model, grouped by key.
    """
    grouped = final_data.groupby(key)
    results = []

    for group_name, group_df in grouped:
        result_row = {key: group_name}

        for model in model_list:
            forecast_col = f"forecast_{model}"

            # MAPE Calculation (handle division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                percentage_errors = np.abs(group_df["target"] - group_df[forecast_col]) / group_df["target"]
                percentage_errors = percentage_errors.replace([np.inf, -np.inf], np.nan).dropna()
                mape = percentage_errors.mean() * 100 if not percentage_errors.empty else np.nan

            result_row[model] = mape

            # Mean Bias Calculation
            bias = (group_df[forecast_col] - group_df["target"]).mean()
            result_row[f"bias_{model}"] = bias

        results.append(result_row)

    return pd.DataFrame(results)
