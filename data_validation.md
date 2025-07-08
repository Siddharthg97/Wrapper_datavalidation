## Data Validation checks are to performed in 
1) **check data type**
**isinstance(object, classinfo)**

2)  **if values/columns is already created then avoid re-creating that**
   

object: Any value or variable you want to examine.
classinfo: A type (like int, str, list, etc.), a custom class, or even a tuple of types.
  
isinstance(5, int)                # True
isinstance("Hi", str)             # True
isinstance([1,2,3], (list, tuple))# True for either type
class Animal: pass
class Dog(Animal): pass

dog = Dog()
isinstance(dog, Dog)    # True
isinstance(dog, Animal) # True — checks subclass too :contentReference[oaicite:7]{index=7}

**Error**
ZeroDivisionError – dividing by zero

NameError – referring to an undefined variable

TypeError – using incorrect types (e.g., '2' + 2)

ValueError – correct type but invalid value (e.g., int('abc'))

IndexError – accessing out-of-range index

KeyError – accessing missing dictionary key

AttributeError – accessing a non-existent attribute

ImportError/ModuleNotFoundError – import issues

FileNotFoundError, MemoryError, StopIteration, RuntimeError, RecursionError, etc.

**Key Error**
If you’re writing a function where a missing key signifies invalid input or corrupted data, it's fine to do:
if 'name' not in data:
    raise KeyError("'name' is required")

# Ensure date is datetime
if date_column not in df_month.columns:
    raise KeyError(f"Date column '{date_column}' not found.")
if not pd.api.types.is_datetime64_any_dtype(df_month[date_column]):
    df_month[date_column] = pd.to_datetime(df_month[date_column])

  **Value Error**
A ValueError in Python is raised when a function or operation receives an argument of the correct type, but the value itself is invalid or inappropriate for that operation
def calculate_area(length, width):
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive numbers.")
    return length * width

**run time error**


**try & exception**
except Exception as e:
    # handle error...
…it means you're catching all exceptions that inherit from the base Exception class (which includes most runtime errors)
and storing the error inside the variable e. You can then use **str(e) or e.args** to read the specific error message

try block runs your code.

If any Exception (or subclass) is raised, control jumps to the except block.

The exception instance is assigned to e.

result = 10 / 0
except Exception as e:
    print(f"Error occurred: {e}")
    # e is a ZeroDivisionError: "division by zero"

except (ZeroDivisionError, ValueError) as e:
    # handle known errors
Catch broad only when needed, e.g., in top-level loops to prevent crashing, but always log and re-raise if necessary .

If using except Exception as e, log the full stack trace:

import logging
logging.exception("Something went wrong")
This provides context for debugging

except Exception as e: catches most runtime errors and gives you access to the error message.
⚠️ Avoid overusing it — catching only what you expect is safer and makes maintenance easier.
When used, log properly, handle cleanup, and optionally re-raise to preserve tracebacks.
Let me know if you'd like examples for data processing, network calls, or any other scenario!

try:
    ...
except Exception as e:
    handle_error(e)
    
def handle_error(e):
    logger.error("Error occurred", exc_info=e)
    cleanup()
    # maybe re-raise or set fallback values

BaseException
 ├─ GeneratorExit
 ├─ KeyboardInterrupt
 ├─ SystemExit
 └─ Exception
      ├─ ArithmeticError (ZeroDivisionError, OverflowError…)
      ├─ LookupError (IndexError, KeyError…)
      ├─ ValueError, TypeError, OSError, etc.

Use Exception for most error handling (covers typical runtime errors).
BaseException includes critical exceptions like KeyboardInterrupt and SystemExit that you usually don’t 
want to catch, unless you have a very specific reason.

except Exception as e:
    # If an error occurs, capture the error message and set forecast to NaN
    error_str = f"{df[base_vars['key']].unique()[0]} Exception: {str(e)}"
    dff['run_status_' + model_name] = error_str
    dff['forecast'] = np.nan

**logging**
***System arguments***
if __name__ == '__main__':
    args = json.loads(sys.argv[1]) 
    project_id = args[""] 
    bq_dataset = args[""] 
    ds_temp_bucket = args[""]
    source_dataset = args[""]


***Get or Create the Logger***

logging.getLogger('py4j'): Creates or retrieves a logger named 'py4j'.
2)Set Logging Level:
  logger.setLevel(logging.INFO): Configures the logger to handle messages at the INFO level and above 
  (INFO, WARNING, ERROR, CRITICAL).

3)Create a Console Handler:
  ch = logging.StreamHandler(): Creates a handler that outputs log messages to the console.
  ch.setLevel(logging.INFO): Sets the console handler to only process messages at the INFO level or above.

4)Create a Formatter:
  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'): Specifies the format for log messages, including:
  %(asctime)s: Timestamp of the log entry.
  %(name)s: Name of the logger ('py4j' in this case).
  %(levelname)s: Log level (INFO, ERROR, etc.).
  %(message)s: The actual log message.
  Attach Formatter to the Handler:

5)ch.setFormatter(formatter): Adds the formatter to the console handler.
6)Add Handler to the Logger:
  logger.addHandler(ch): Adds the console handler to the logger, enabling the formatted log messages to be
  displayed on the console.

Input code
  logger.info("This is an info message.")
  logger.warning("This is a warning message.")
  logger.error("This is an error message.")


Output code
2024-12-24 12:00:00 - py4j - INFO - This is an info message.
2024-12-24 12:00:01 - py4j - WARNING - This is a warning message.
2024-12-24 12:00:02 - py4j - ERROR - This is an error message.

**handle inf values**

**run time calculation**
%%time



**How to write a function**
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
  
