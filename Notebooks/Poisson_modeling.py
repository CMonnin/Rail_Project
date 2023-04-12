## pipeline creator imports
# the model imports
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
# preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer 
from sklearn.preprocessing import Normalizer
# other imports
import pandas as pd

## imports for regression results and pipeline functions
import sklearn.metrics as metrics

## regression results function imports
import numpy as np

## imports for plotter function
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_validate
######################################################################


class Poisson_modeling:
    
    def __init__(self,
                 data = pd.read_csv('../Data/Group_by.csv',low_memory=False)):

        self.all_features = [
        'Trains Daily',
        'Vehicles Daily',
        'Train Max Speed (mph)',
        'Road Max Speed (km/h)',
        'Province', 
        'Protection'
        ]
        # Cat features of interest
        self.categorical_features = [
            'Province', 
            'Protection'
            ]
        # Numerical features of interest
        self.numeric_features = [
            'Trains Daily',
            'Vehicles Daily',
            'Train Max Speed (mph)',
            'Road Max Speed (km/h)',
            ]
        self.X = data[self.all_features]
        self.y = data['Count']


    # Delete this function
    def preprocessor(self,
                     input_scaler):
        all_features = self.all_features
        categorical_features = self.categorical_features
        numeric_features = self.numeric_features
        X = self.X        
        y = self.y
        # OneHotEncoder to get dummy variables for the cat variables
        categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ]
        )
        # Apply a scaler to the numerical features
        numeric_transformer = Pipeline(
            [
                ('scaler', input_scaler)
            ]
        )

        preprocessor = ColumnTransformer(
            # do the processing on these features of interest and drop the remainders
            # there shouldn't be any remainders but it's a nice feature to have
            [
                ('categoricals', categorical_transformer, categorical_features),
                ('numericals', numeric_transformer, numeric_features)
            ],
            remainder = 'drop'
        )
        # Creating the pipeline 
        pipeline = Pipeline(
            [
                ('preprocessing', preprocessor),
            ]
        )
        return preprocessor


    def pipeline_creator(
            self,             
            input_est=PoissonRegressor(),
            input_scaler =StandardScaler(),
            test_train=True
            ):
        '''Function is specific to the data set and features are currently hardcoded in
        returns a pipeline
        '''
        all_features = self.all_features
        categorical_features = self.categorical_features
        numeric_features = self.numeric_features
        X = self.X        
        y = self.y

        # OneHotEncoder to get dummy variables for the cat variables
        categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ]
        )
        # Apply a scaler to the numerical features
        numeric_transformer = Pipeline(
            [
                ('scaler', input_scaler)
            ]
        )

        preprocessor = ColumnTransformer(
            # do the processing on these features of interest and drop the remainders
            # there shouldn't be any remainders but it's a nice feature to have
            [
                ('categoricals', categorical_transformer, categorical_features),
                ('numericals', numeric_transformer, numeric_features)
            ],
            remainder = 'drop'
        )

        # Creating the pipeline 
        pipeline = Pipeline(
            [
                ('preprocessing', preprocessor),
                ('est', input_est )
            ]
        )
        # if a test_train has been opted for 
        if test_train == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            print('**************************')
            print(f'est type: {input_est}')
            print(f'scaler type: {input_scaler}')
            print(f'Test_train: {test_train}')
            self.regression_results(y_test,y_pred)
            return pipeline
            
        else:
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            print('**************************')
            print(f'est type: {input_est}')
            print(f'scaler type: {input_scaler}')
            print(f'Test_train: {test_train}')
            print('***')
            self.regression_results(y,y_pred)
            return pipeline
    
    def regression_results(self,y,y_pred):
        '''attempt at emulating stats model summary by 
        reporting various metrics of interest from SKlearn models'''
        # Regression metrics
        mask = y_pred > 0
        explained_variance=metrics.explained_variance_score(y, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y, y_pred) 
        mse=metrics.mean_squared_error(y, y_pred) 
        mean_squared_log_error=metrics.mean_squared_log_error(y, y_pred)
        median_absolute_error=metrics.median_absolute_error(y, y_pred)
        r2=metrics.r2_score(y, y_pred)
        mean_poisson_deviance=metrics.mean_poisson_deviance(y[mask], y_pred[mask])

        print('explained_variance: ', round(explained_variance,4))
        print('RMSE: ', round(np.sqrt(mse),4))
        print('mean poisson_deviance',round(mean_poisson_deviance),4)    
        print('***')
        print('mean_squared_log_error: ', round(mean_squared_log_error,4))
        print('r2: ', round(r2,4))
        print('MAE: ', round(mean_absolute_error,4))
        print('MSE: ', round(mse,4))
        print('Median absolute error: ', round(median_absolute_error,4))
    

    def evaluate_model(self,est, model, n_jobs=1):
        # adapted from:  https://github.com/thomasjpfan/scipy-2022-poisson
        X = self.X
        y = self.y
        custom_scorers = {
            'explained_variance':make_scorer(metrics.explained_variance_score),
            'mean_possion_devianace': make_scorer(metrics.mean_poisson_deviance),
            'root_mean_squared_error': make_scorer(metrics.mean_squared_error),
        }
        results = cross_validate(est, X, y, scoring=custom_scorers, cv=100, n_jobs=n_jobs, return_estimator=True)
        output = {"model": model}
        for key, values in results.items():

            # print(f'key: {key}')
            # print(f'value: {values}')
            # print('here')
            if key == "estimator":
                continue
            key_name = key[5:]
            output[key_name] = results[key]
            output[f"{key_name}_mean"] = results[key].mean()
        output["estimator"] = results["estimator"]
        return output




    def plot_results(self,*results):
    # adapted from:  https://github.com/thomasjpfan/scipy-2022-poisson
        metrics = ['explained_variance',"root_mean_squared_error", "mean_possion_devianace"]
        title = ['Explained Variance',"Root Mean Squared Error", "Mean Possion Deviance"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4), sharey=True, dpi=300, constrained_layout=True)
        for i, metric in enumerate(metrics):
            dfs = [pd.DataFrame({
                metric: result[metric],
                "model": result["model"],
            }) for result in results]
            df_all = pd.concat(dfs)
        
            ax = axes[i]
            sns.swarmplot(y="model", x=metric, data=df_all, ax=ax, color=".25")
            sns.boxplot(y="model", x=metric, data=df_all, ax=ax)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(title[i], fontsize=18)
            ax.tick_params(axis="y", labelsize=16)
        return fig



    def save_results(self,*results):
        metrics = ['explained_variance',"root_mean_squared_error", "mean_poisson_deviance"]
        for i, metric in enumerate(metrics):
            dfs = [pd.DataFrame({
                metric: result[metric],
                "model": result["model"],
            }) for result in results]
            model = results[0].get('model')
            path = '../Data/'+str(model)+'_'+str(metric)+'_plot_data.csv'
            dfs[0].to_csv(path)
    

    def plotter_from_file(self,filename):
        df = pd.read_csv(filename)
        fig = plt.figure(figsize=(9,3))
        metric = df.iloc[:,1]
        model = df.iloc[:,2]
        sns.swarmplot(y=model, x=metric, data=df, color=".25")
        sns.boxplot(y=model, x=metric, data=df)  
        return fig


