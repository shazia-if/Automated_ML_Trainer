import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


#This one is the final version

header = st.container()
dataset = st.container()
feature_selection = st.container()
model_train = st.container()
model_performance = st.container()
predictions = st.container()


@st.cache_data
def get_dataset(filepath):
    data = pd.read_csv(filepath)
    return(data)
    
@st.cache_data
def drop_cols(columns_to_drop):
    data = df.drop(columns=columns_to_drop)
    return data


def feature_select(features):
    data= df[features]
    return data




with header:
    st.title("Automated ML Trainer")
    st.text("This Machine Learning Trainer is a basic model designed to automatically train on user-provided datasets and generate predictive outputs.")


with dataset:
    st.header("Dataset selection")
    st.text("Please upload a CSV data file to begin your model analysis.")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = get_dataset(uploaded_file)
            # Check if the dataset is empty
            if df.empty:
                st.error("The uploaded dataset is empty. Please upload a valid CSV file.")
            # Check for at least one numerical column
            elif df.select_dtypes(include=['int', 'float']).shape[1] == 0:
                st.error("The dataset must contain at least one numerical column for analysis.")
            else:
                st.write("Preview of the dataset:")
                st.write(df.head())

                columns_to_drop = df.select_dtypes(exclude=['int', 'float'])
                if st.button("Drop Columns"):
                    st.text("All non-integer and non-float columns will be excluded to enable a simplified analysis by the model.")
                    df = drop_cols(columns_to_drop)

                    """storing datafreame in session state"""
                    if "data" not in st.session_state:
                     st.session_state["data"] = df
                    st.success(f"Dropped columns: {columns_to_drop}")

                df= st.session_state.data


        except:
            pass

    else:
        st.write("Please choose a csv file first")


with feature_selection:
    st.header("Feature Selection")
    st.text("In this section, you can select the target variable and one feature at a time for analysis. This allows you to compare and determine which feature is most relevant.")

    try:
        # Show updated dataset
        st.subheader("Updated Dataset")
        st.write(df)

        st.markdown("""
        **Note:** The target variable (y) is the column you want to predict.  
        For example:
        - If predicting house prices, select the 'Price' column.
        - If classifying emails as spam or not, select 'Spam' (1/0).
        """)

        is_categorical = st.radio("Is the target variable categorical?", ("Yes", "No"))

        target_column = st.selectbox("Select the target column (y):", df.columns)
        st.write(f"You selected `{target_column}` as the target variable.")

        feature_cols= st.selectbox("Select your model's feature:", df.columns)
        if st.button("Select a feature for analysis "):
            selected_features = feature_select(feature_cols)
            if "features" not in st.session_state:
                st.session_state["features"] = selected_features

            if "target" not in st.session_state:
                st.session_state["target"] = df[target_column]

            X = selected_features  # Features
            y = df[target_column]  # Target variable

                # Store and display
            st.write("The selected feature column:")
            st.write(X)
            st.write("The target variable column:")
            st.write(y)

        

       

    except NameError:
        pass
    except Exception as e:
        st.write(f"error: {e}")



with model_train:
    st.header("Model Training")
    st.text("In this section, the model will be trained using the selected feature to predict the target variable. The performance of multiple models will be evaluated and presented for comparison. For **regression** tasks, the models used are Random Forest Regressor, Decision Tree Regressor, and Linear Regression. For **classification** tasks, the models used are Random Forest Classifier, Logistic Regression, and Naïve Bayes.")

    if st.button("Train Models"):
        try:
            X = st.session_state.features
            if isinstance(X, pd.Series):
                X = X.to_frame()

            y = st.session_state.target
            if isinstance(y, pd.Series):
                y = y.to_frame().squeeze() 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            if "X_training" not in st.session_state:
                st.session_state["X_training"]= X_train

            if "y_train" not in st.session_state:
                st.session_state["y_train"] = y_train

            if is_categorical == "No":  # Regression models
                models = {
                            'Random Forest': RandomForestRegressor(),
                            'Decision Tree': DecisionTreeRegressor(),
                            'Linear Regression': LinearRegression(),
                        }
            else:  # Classification models
                models = {
                            'Random Forest': RandomForestClassifier(),
                            'Logistic Regression': LogisticRegression(),
                            'Naïve Bayes': GaussianNB(),
                        }
            
            if "models_list" not in st.session_state:
                st.session_state["models_list"] = models

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if is_categorical == "No":
                    results[name] = {
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "MSE": mean_squared_error(y_test, y_pred),
                            }
                else:
                    results[name] = {"Accuracy": accuracy_score(y_test, y_pred)}

            
            if "results" not in st.session_state:
                st.session_state["results"]= results


            
            st.write("Model Evaluation:")
            st.json(results)

        except NameError:
                pass
        except ValueError:
                st.subheader("It looks like you didn't choose the type of target variable correctly!")
                pass




with model_performance:
    st.header("Model Performance")
    st.text("The top-performing model will be chosen here to generate further predictions. You can modify the inputs to explore different predictions made by the model.")

    if st.button("Pick best model"):
        

        try:
            results = st.session_state["results"]
            
            if is_categorical =="No":
                # Finding the model with the lowest MSE and informing the user
                best_model = min(results, key=lambda model: results[model]['MSE'])
            
            else:
                best_model = min(results, key=lambda model: results[model]['Accuracy'])

            if "best model" not in st.session_state:
                st.session_state["best model"]= best_model

            st.subheader(f"The best Performing model is: {best_model}")
            st.text("This is the model with lowest error or highest accuracy")
        except NameError:
            st.write("There was an error")
        except Exception as e:
            st.text(f"couldn't choose best model due to the following: {e}")
            pass

with predictions:
    st.subheader("Provide input for prediction")
   
        
    try:    
        # Get the feature DataFrame from session_state
        features_df = pd.DataFrame(st.session_state["features"])
        column = features_df.columns[0]

            # Get min and max for the only feature column
        min_val = features_df[column].min()
        max_val = features_df[column].max()

            # Take a single input from user within the column's range
        user_input = st.number_input(
            f"Enter value for '{column}'",
            min_value=float(min_val),
            max_value=float(max_val),
            key="user_input"
            )

        if "user_input" not in st.session_state:
            st.session_state["user_input"]= user_input

        if st.button("done"):

            st.write(f"Input value: {user_input}")


    except Exception as e:
                    st.text(f"We have received this Error:{e}")



if st.button("Predict"):
                        
                    user_input = st.session_state["user_input"]
                    if is_categorical == "No":  # Regression models
                        models = {
                                                'Random Forest': RandomForestRegressor(),
                                                'Decision Tree': DecisionTreeRegressor(),
                                                'Linear Regression': LinearRegression(),
                                            }
                    else:  # Classification models
                        models = {
                                                'Random Forest': RandomForestClassifier(),
                                                'Logistic Regression': LogisticRegression(),
                                                'Naïve Bayes': GaussianNB(),
                                            }   
                            
                    features_df = st.session_state["features"]
                    features_df = pd.DataFrame(features_df)
                    column = features_df.columns[0] 
                    unique_vals = features_df[column].unique()
                            
                    best_model = st.session_state["best model"]
                    model= models[best_model]
                    
                    st.text(user_input)
                    user_input_given = [[user_input]]

                    X_train = st.session_state["X_training"]
                    y_train = st.session_state["y_train"]
                    if isinstance(X_train, pd.Series):
                        X_train = X_train.to_frame()

                    if isinstance(y_train, pd.Series):
                        y_train = y_train.to_frame().squeeze()

                    model.fit(X_train, y_train)
                    prediction = model.predict(user_input_given)
             
                    st.success(f"The model has predicted the following output: {prediction[0]}")











    