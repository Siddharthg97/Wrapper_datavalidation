def random_forest(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_parameters: dict,
    tuning: bool = True,
    key:str = 'Test',
    inference_flag:bool = False,
    verbose: bool = False
):
    """
    random_forest model - multivariate model with two options .True for hyperparameter tuning and False for fixed values for parameters.
    Input is a dataframe with target column (independent variable) and features created from create_features() function.
    Args
        :param x_train: Training dataset with the regressors
        :param y_train: Target data
        :param x_test: Testing dataset with the regressors
        :param model_parameters: dictionary of parameters
        :param tuning: hyperparameter setting
    Returns:
        list containing the forecast values
    """
    if tuning==True:
        params = model_parameters["True"]
        reg = parameter_tuning(
            x_train = x_train,
            y_train = y_train,
            regressor = RandomForestRegressor(),
            params = params,
            n_iter = 10
        )
        model = reg.best_estimator_
        model.fit(x_train,y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)
        return model, pred
    else:
        params = model_parameters["False"]
        n_estimators = params["n_estimators"]
        max_depth = params["max_depth"]
        model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
        )
        model.fit(x_train,y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)
        return model, pred

def xgboost(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_parameters: dict,
    tuning: bool = True,
    key:str = 'Test',
    inference_flag:bool = False,
    verbose: bool = False
):
    """
    xg_boost model - multivariate model with two options .True for hyperparameter tuning and False for fixed values for parameters.
    """
    # Replace infinite values with NaN
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)
    
    if tuning == True:
        params = model_parameters["True"].copy()
        params['missing'] = np.nan  # Ensure missing is set to handle NaN
        
        reg = parameter_tuning(
            x_train = x_train,
            y_train = y_train,
            regressor = XGBRegressor(enable_categorical=True, missing=np.nan),
            params = params,
            n_iter = 10
        )
        model = reg.best_estimator_
        model.fit(x_train, y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)
    else:
        params = model_parameters["False"].copy()
        params['missing'] = np.nan  # Ensure missing is set to handle NaN
        
        model = XGBRegressor(
            enable_categorical=True,
            missing=np.nan,
            n_estimators = params["n_estimators"],
            learning_rate = params["learning_rate"],
            max_depth = params["max_depth"],
            n_jobs = params["n_jobs"],
            alpha = params["alpha"],
            tree_method = params.get("tree_method", 'hist')
        )
        model.fit(x_train, y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)

    return model, pred

def lgbm(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    model_parameters: dict,
    tuning: bool = True,
    key:str = 'Test',
    inference_flag:bool = False,
    verbose: bool = False
):
    """
    LightGBM forecast model
    """
    # Replace infinite values with NaN
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)
    
    if tuning == True:
        params = model_parameters["True"].copy()
        params.pop('verbose', None)
        reg = parameter_tuning(
            x_train = x_train,
            y_train = y_train,
            regressor = LGBMRegressor(),
            params = params,
            n_iter = 10
        )
        model = reg.best_estimator_
        model.fit(x_train,y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)
    else:
        params = model_parameters["False"]
        model = LGBMRegressor(
            objective = params["objective"],
            boosting_type = params["boosting_type"],
            num_leaves= params["num_leaves"],
            learning_rate = params["learning_rate"],
            force_col_wise = params["force_col_wise"],
            verbose= params["verbose"],
            seed= params["seed"]
        )
        model.fit(x_train,y_train)
        if inference_flag == True:
            return model
        pred = model.predict(x_test)
    return model, pred
---------------------------------------- baseline model-----------------
#  models
model_xgb,xgb_pred=xgboost(
    x_train,
    y_train,
    x_test,
    model_parameters_dict["xgboost"],
    False
)


model_lgbm,lgm_pred=lgbm(
    x_train,
    y_train,
    x_test,
    model_parameters_dict["lgbm"],
    False
)

model_rf,rf_pred=random_forest(
    x_train,
    y_train,
    x_test,
    model_parameters_dict["random_forest"],
    False
)

--------------------------- save the model ------------------------------------------------


local_model_dir="saved_models/tuned"
os.makedirs(local_model_dir, exist_ok=True)

for mdl_obj,mdl_name  in zip([model_xgb,model_lgbm,model_rf],["xgboost","lightgbm","random_forest"]):
    model_package=(mdl_obj,expected_features)
    local_model_path = os.path.join(local_model_dir, f"{mdl_name}_model.pkl")
    with open(local_model_path, 'wb') as f:
                pickle.dump(model_package, f)

-------------------------- open the model ---------------------------------------------------

local_model_dir="saved_models/tuned"
model_name="xgboost"
local_model_path = os.path.join(local_model_dir, f"{model_name}_model.pkl")
with open(local_model_path, 'rb') as f:
    model1 = pickle.load(f)
xgb_model=model1[0]
expected_features=model1[1]

--------------------- all 3 models training/tuning & saving model instance --------------------
multivariate_model_list = ['lgbm', 'xgboost', 'random_forest']

model_test = {
    'lgbm': lgbm,
    'xgboost': xgboost,
    'random_forest': random_forest
}



model_parameters_dict = {
    "lgbm": {
        "True": {
            'objective': ['regression'],
            'metric': ['mse'],
            'boosting_type': ['gbdt'],
            'num_leaves': [25,30,35,40],
            'learning_rate': [0.01,0.02,0.05,0.07],
            'feature_fraction': [0.5,0.7,0.9],
            'bagging_fraction': [0.7,0.75,0.8],
            'bagging_freq': [2,5,7],
            'verbose': 0,
            'seed': [42]
        },
        "False": {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 4,
            'learning_rate': 0.05,
            'force_col_wise': True,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 42
        }
    },
    "xgboost": {
        "True": {
            "max_depth": list(range(3, 10 + 1)),
            "learning_rate": [0.3, 0.2, 0.1, 0.05],
            "tree_method": ['hist'],
            "missing": [np.nan],  # Explicitly handle missing values
            "n_estimators": list(range(100, 300 + 1, 20)),
            "gamma": [0, 0.1, 0.01, 0.001],
            "subsample": [0.9, 0.8, 0.7],
            "colsample_bytree": [0.9, 0.8, 0.7],
            "colsample_bylevel": [0.9, 0.8, 0.7],
            "scale_pos_weight": [0.9, 0.8, 0.7],
            "importance_type": ['gain', 'weight', 'cover', 'total_gain', 'total_cover']
        },
        "False": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_jobs": 1,
            "alpha": 0.1,
            "tree_method": 'hist',
            "missing": np.nan  # Explicitly handle missing values
        }
    },
    "random_forest": {
        "True": {
            "n_estimators": list(range(100, 300 + 1, 10)),
            "max_depth": list(range(3, 15 + 1)),
            "min_samples_split": [2, 3, 4, 5, 6],
            "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],
            "max_features": [None, 'sqrt', 'log2', 0.9, 0.85, 0.75],
            "bootstrap": [True, False]
        },
        "False": {
            "n_estimators": 50,
            "max_depth": 200,
        }
    }
}


def parameter_tuning(x_train, y_train, regressor, params, n_iter=10):
    """
    Perform RandomizedSearchCV for hyperparameter tuning with proper parameter handling
    """
    # Replace infinite values if not already done
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    
    # Make a copy of params to avoid modifying the original
    search_params = params.copy()
    
    # For XGBoost, ensure missing parameter is properly formatted
    if isinstance(regressor, XGBRegressor):
        if 'missing' in search_params:
            # If missing is specified as a single value, wrap it in a list
            if not isinstance(search_params['missing'], (list, np.ndarray)):
                search_params['missing'] = [search_params['missing']]
    
    search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=search_params,
        scoring='neg_root_mean_squared_error',
        n_iter=n_iter,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(x_train, y_train)
    return search
-----------------------------------------------------------------------------------------------------------------------------------------------------------
def multivariate_model_sprint(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of multivariate forecasting models to grouped data and return the combined results.

    This function iterates through a list of multivariate forecasting models (such as XGBoost and Random Forest), 
    applies each model to the input data, and aggregates the forecast results.

    Parameters:
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data for forecasting, 
        which includes columns such as 'date', 'key', and other necessary features for the models.

    Returns:
    -------
    pandas.DataFrame
        DataFrame with columns for date, key, forecasted values from each model, 
        and the original data. Missing values in forecast columns are replaced with infinity.

    Note:
    -----
    - The function assumes that `multivariate_model_list` is a list of model functions 
      and `model_calling` is a function that takes a model and a DataFrame to generate forecasts.
    - The date column is parsed to datetime format, and forecasts are combined with actual data for each key.
    - The `create_features` function is used to generate the necessary features for the models based on the input data.
    """

    # Prepare the input DataFrame and parse the 'date' column to datetime format
    # df_input = data.copy()  # Create a copy of the input data to avoid modifying it directly
    print('multivariate_model_sprint')
    
    df_input[base_vars['date']] = pd.to_datetime(df_input[base_vars['date']], errors='coerce')

    # Create features necessary for input to multivariate models
    df = df_input

    # Initialize an empty DataFrame to hold the final results
    final = pd.DataFrame()

    # Loop through each model function in the multivariate model list
    for model_name in multivariate_model_list:
        model_func = model_test[model_name]
        # Initialize a DataFrame to hold the results for the current model
        master = pd.DataFrame(columns=[base_vars['date'], base_vars['key'], 'forecast'])

        # Subset data for the current key (only one key is used here)
        df_key = df  # If using as pandas, you might want to filter for specific keys here

        # Split data into training and testing sets
        X_train_1 = df_key.iloc[:-horizon, :].drop(base_vars['target'], axis=1)  # Training features
        X_test_1 = df_key.drop(base_vars['target'], axis=1)  # Test features
        Y_train_1 = df_key.iloc[:-horizon, :][[base_vars['key'], base_vars['target']]]  # Training target
        Y_test_1 = df_key[[base_vars['key'], base_vars['target']]]  # Test target

        # Create a temporary DataFrame to hold the forecast results
        dff = pd.DataFrame(columns=[base_vars['date'], base_vars['key'], 'forecast'])
        dff[base_vars['date']] = X_test_1[base_vars['date']]  # Copy date column
        dff[base_vars['key']] = df[base_vars['key']].unique()[0]  # Set the key value

        # Try-except block to handle potential errors during model training/prediction
        try:
            if model_name == 'fb_prophet':
                x_test=X_test_1.drop([base_vars['key']], axis=1)
              # Call the current model function to generate forecasts
                dff['forecast'] = model_func(
                    x_train=X_train_1.drop([base_vars['key']], axis=1),
                    y_train=Y_train_1.drop([base_vars['key']], axis=1),
                    x_test=X_test_1.drop([base_vars['key']], axis=1),
                    model_parameters=model_parameters_dict[model_name],  # Model-specific parameters
                    tuning=True,  # Tuning flag for hyperparameters
                    key=df[base_vars['key']].unique()[0]  # Current key
                )
            else:
                # Call the current model function to generate forecasts
                model,dff['forecast'] = model_func(
                    x_train=X_train_1.drop([base_vars['key'], base_vars['date']], axis=1),
                    y_train=Y_train_1.drop([base_vars['key']], axis=1),
                    x_test=X_test_1.drop([base_vars['key'], base_vars['date']], axis=1),
                    model_parameters=model_parameters_dict[model_name],  # Model-specific parameters
                    tuning=True,  # Tuning flag for hyperparameters
                    key=df[base_vars['key']].unique()[0]  # Current key
                )
                key = df[base_vars['key']].unique()[0]
                expected_features = X_train_1.drop([base_vars['key'], base_vars['date']], axis=1).columns.tolist()
                model_package = (model, expected_features)
            # Record the status as success
            dff['run_status_' + model_name] = 'success'
            local_model_dir = "saved_models"
            os.makedirs(local_model_dir, exist_ok=True)
            local_model_path = os.path.join(local_model_dir, f"{key}_{model_name}_model.pkl")
            with open(local_model_path, 'wb') as f:
                pickle.dump(model_package, f)
        except Exception as e:
            # If an error occurs, capture the error message and set forecast to NaN
            error_str = f"{df[base_vars['key']].unique()[0]} Exception: {str(e)}"
            dff['run_status_' + model_name] = error_str
            dff['forecast'] = np.nan

        # Concatenate the current model results to the master DataFrame
        master = pd.concat([master, dff], axis=0)

        # Drop rows with NaN in the key column to clean up the results
        master.dropna(subset=[base_vars['key']], inplace=True)

        # Rename the forecast column to include the model name
        master.rename(columns={'forecast': f'forecast_{model_name}'}, inplace=True)

        # Ensure the date column is in datetime format
        master[base_vars['date']] = pd.to_datetime(master[base_vars['date']])
        # master.to_csv('inter_out.csv',index=False)

        # Merge the results into the final DataFrame
        if final.empty:
            final = master.copy()  # If final is empty, initialize it with master
        else:
            final = pd.merge(final, master, on=[base_vars['key'], base_vars['date']], how='inner')  # Merge

    # Merge the predicted values with the actual data
    final = pd.merge(final, df_input, on=[base_vars['key'], base_vars['date']], how='left')

    # Replace any NaN values in the final DataFrame with infinity
    final.fillna(np.inf, inplace=True)

    return final  # Return the combined results DataFrame

------------------------------------------------------------------------------ call the function-------------------------------------------------------------------------------

    %%time
mult_future = pd.concat(
        [multivariate_model_sprint(group) for key_value, group in df.groupby('key')],
        ignore_index=True
    )
    
--------------------------------------------------- metrics-----------------------------



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
    
# Calculate WMAPE
def weighted_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

def results(df_res,actuals:str,forecast:str,model_name:str,key:str,date:str):
    """ metrics - mae, rmse, r2, std_dev are obtained on predicted values
        Plot of target v/s forecasted is obtained on test data 
    """
    df_temp=df_res.copy()
    y_test=df_temp[actuals]
    y_pred=df_temp[forecast]
    print(f"---------{model_name}------------")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    std_dev = y_test.std()
    rmse_std_ratio = rmse / std_dev

    wmape = weighted_mean_absolute_percentage_error(y_test, y_pred)

    # Output metrics
    print("\nModel Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Standard Deviation of Test Data: {std_dev:.2f}")
    print(f"RMSE/Std Dev Ratio: {rmse_std_ratio:.2f}")
    print(f"WMAPE: {wmape:.2f}%")

    # # Plot feature importance
    # plt.figure(figsize=(10, 6))
    # xgb.plot_importance(xgb_model, max_num_features=20)
    # plt.title('XGBoost Feature Importance')
    # plt.tight_layout()
    # plt.show()
    index_plot=df_temp[key]+"_"+\
    df_temp[date].astype(str).str.replace(r'-\d{2}$', '', regex=True)
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual', linewidth=2)
    plt.plot(range(len(y_test)) ,y_pred, label='Predicted', linestyle='--', linewidth=2)
    plt.title(f'Monthly Store transactions for April & May Forecast:({model_name})')
    plt.xticks(range(len(y_test)),index_plot)
    plt.xlabel('Store_Month')
    plt.ylabel('Transaction count')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.show()

results(apr_may_df,"target","forecast","XGboost","key","date")
