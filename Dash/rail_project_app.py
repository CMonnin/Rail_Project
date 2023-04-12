
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
######################
# data used for plotly

# set the theme to a simialar seaborn theme
pio.templates.default = "simple_white"
EDA_data = pd.read_csv('https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/Group_by.csv')

plot_data = {
    'metric':['forest','forest','forest','HistGradientBoostingRegressor','HistGradientBoostingRegressor','HistGradientBoostingRegressor','PoissonRegressor','PoissonRegressor','PoissonRegressor'],
    'model':['explained_variance','mean_possion_devianace','root_mean_squared_error','explained_variance','mean_possion_devianace','root_mean_squared_error','explained_varianc','mean_possion_devianace','root_mean_squared_error'],
    'path':['https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/Forest_explained_variance_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/Forest_mean_possion_devianace_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/Forest_root_mean_squared_error_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/HGBR_explained_variance_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/HGBR_mean_possion_devianace_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/HGBR_root_mean_squared_error_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/PR_explained_variance_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/PR_mean_possion_devianace_plot_data.csv',
    'https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Data/PR_root_mean_squared_error_plot_data.csv']
} 

df_plot = pd.DataFrame(plot_data)

model_results_comparison= [
    {'Model':'Poisson Regressor','Explained_variance':'0.03','RMSE':'0.36','Poisson_Deviance/mse_loss':'0.84'},
    {'Model':'Forest','Explained_variance':'0.01','RMSE':'0.40','Poisson_Deviance/mse_loss':'0.74'},
    {'Model':'HGBR','Explained_variance':'0.10','RMSE':'0.34','Poisson_Deviance/mse_loss':'0.70'},
    {'Model':'Feed-forward NN','Explained_variance':'n/a','RMSE':'0.67','Poisson_Deviance/mse_loss':'0.44'},
    ]
columns = [{'name': col, 'id': col} for col in model_results_comparison[0].keys()]
###################################
# features of interest and some processing of categorical variables
numeric_features = [
    'Count',
    'Trains Daily',
    'Vehicles Daily',
    'Train Max Speed (mph)',
    'Road Max Speed (km/h)',
    ]
categorical_features = [
    'Province', 
    'Protection',
    'Railway'
    ]
EDA_data['Protection_en'] = LabelEncoder().fit_transform(np.asarray(EDA_data['Protection']))
EDA_data['Province_en'] = LabelEncoder().fit_transform(np.asarray(EDA_data['Province']))
EDA_data['Railway_en'] = LabelEncoder().fit_transform(np.asarray(EDA_data['Railway']))
all_features=[
    'Count',
    'Trains Daily',
    'Vehicles Daily',
    'Train Max Speed (mph)',
    'Road Max Speed (km/h)',
    'Province_en', 
    'Protection_en',
    'Railway_en' 
]
all_features2=[
    'Count',
    'Trains Daily',
    'Vehicles Daily',
    'Train Max Speed (mph)',
    'Road Max Speed (km/h)',
    'Province', 
    'Protection',
    'Railway' 
]
scaler_list = ['None',
               'StandardScaler()', 
                  'MinMaxScaler()',
                  'MaxAbsScaler()',
                  'RobustScaler()',
                  "PowerTransformer()",
                  'QuantileTransformer()',
                  'Normalizer()']

# correlation plot for the features chosen 
corr_plot = px.imshow(EDA_data[all_features].corr())

# somme plotting functions
def plotter_from_file(filename):
    df = pd.read_csv(filename)
    metric = df.iloc[:,1]
    model = df.iloc[:,2]
    figure = px.box(df,x=metric,y=model,points="all")
    return figure



footer = html.Div([
    html.Cite('[1] '),
])


################################################################
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1('Train_project'),
            style={'width': 6, 'text-align':'center'}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H3('Objectives'),
                html.P('The original goals of this project were to'),
                html.Ul([
                    html.Li('Find some datasets to use to model train accidents in Canada'),
                    html.Li('Create models both traditional logit models and deep learning models for comparison'),
                    html.Li('Create a dashboard to report findings'),
                    ]),
                html.H3('Data'),
                html.P('Data was obtained from the Transportation Safety Board \
                        of Canada (TSBC) and Transport Canada. \
                        The following csvs were obtained from the TSBC\
                       '),
                html.Ul([
                    html.Li('Occurence: data on the incident.'),
                    html.Li('Train: data on the train involved in the incident'),
                    html.Li('Rolling Stock: information on the rolling stock'),
                    html.Li('Injuries: injuries that occurred'),
                    html.Li('Track and Rolling stock component: information on the components'),
                    ]),
                html.P('After examination of the files and the approx. 700 features present \
                       only the occurrences and train files were used. \
                       The Grade crossing inventory file was downloaded from TC. \
                       This inventory provides information on the rail crossings. \
                       '),
                html.H3('EDA_and_data_wrangling'),
                html.P(children=['The data sets stretch back to 1983. For this project only the last \
                       20 years will be examined. This is due to missing information in the \
                       earlier years. Even when looking at only the last 20 years there is \
                       incomplete information reported. This has been a problem with TSBC \
                       reporting that has been address in the literature [1]. \
                        The Occurrences.csv and Train.csv were merged using Panads. \
                        They were merged on the OCCID feature which was present in both datasets \
                        and was a unique identifier for the accident. \
                        These combined datasets where then merged with the Grade Inventory.csv \
                        with a left join where the Grade Inventory was the left dataset. \
                        This ensured that even the locations where no accident occurred in the last 20 years were \
                        included. \
                       After this examination of merged datasets and features the following features \
                       were chosen',
                       html.Code(html.Ul([html.Li(x) for x in all_features])),
                       'Here ',
                       html.Code('Count'),
                        ' refers to the number of incidents that occurred \
                        at a grade crossing in the last 20 years. The features with the _en suffix \
                        are categorical variables and the rest are numerical variables. ',
                        html.Code('Protection'),
                        ' refers to the type of protection present at the grade crossing. Either active or passive ',
                        html.Code('Railway'),
                        ' refers to the owner of the railway that the grade crossing is part of \
                        next an examination of the selected features will be performed'
                       ]),
                dcc.Dropdown(all_features2,
                            value=all_features2[0],
                            id='EDA-dropdown1'),
                dcc.Graph(id='EDA-plot1'),
                html.P(children=['The plots above show that the counts have a high number of counts at 0 and 1. \
                       This is observed for ',
                       html.Code ("Trains Daily"),
                        ' and ',
                        html.Code( "Vehicles Daily"), 
                        ' too. \
                       The correlation matrix below shows that no the only variables that display a relationship to each \
                       other are "Train max speed" and "Trains Daily" with a correlation score of 0.55. The categorical \
                       variables have been encoded to numpy arrays using sklearn function ',
                       html.Code('labelEncoder()'),
                                 ]),  
                dcc.Graph(id='corr_plot',figure=corr_plot),
            ]),
            style={
            "text-align": "justify",
            "width": "4"}
        ),
        dbc.Col(
            html.Div([
                html.H3('Model_selection'),
                html.H4('Generalised_linear_models'),
                html.P(children = ['As the final dataset is count data a Poisson Regression in a natural choice \
                                   rather than a logit model. Three different generalised linear models (GLM) were assessed. ',
                                   html.Code('PoissonRegressor()'),
                                   '(PR), ',
                                   html.Code('RandomForestRegressor(criterion="poisson")'),
                                   '(Forest), and, ',
                                   html.Code('HistGradientBoostingRegressor(loss="poisson")'),
                                   '(HGBR) were selected as the three estimators to use. These are made \
                                    made available by SKlearn. A number of different scalers were also assessed:',
                                    html.Code((html.Ul([html.Li(x) for x in scaler_list]))),
                                    'Using a class created for this project, ',
                                    html.Code('Poisson_modeling()'),
                                    " pipelines were set up do test the different combinations of preprocessing \
                                    scalers and estimators. The metrics that were given the most importance were: \
                                    Explained Variance, Root Mean Squared Error, and Mean Poisson Deviance \
                                    for each model the best scaler was chosen according to these metrics as a form of tuning. \
                                    PR: StandardScaler(), Forest: None, and HGBR: None. For each of these 3 methods \
                                    the models were evaluated using Sklearn's ",
                                    html.Code('cross_validate()'),
                                    ' each with cv = 100. The model output is a prediction on the number of counts at \
                                    crossing given a set of numerical and categorical features'

                ]),
                dcc.Dropdown(
                    id='model_dropdown',
                    options=[
                        {'label': plot_data['metric'][0]+' '+plot_data['model'][0], 'value':plot_data['path'][0]},
                        {'label': plot_data['metric'][1]+' '+plot_data['model'][1], 'value':plot_data['path'][1]},
                        {'label': plot_data['metric'][2]+' '+plot_data['model'][2], 'value':plot_data['path'][2]},
                        {'label': plot_data['metric'][3]+' '+plot_data['model'][3], 'value':plot_data['path'][3]},
                        {'label': plot_data['metric'][4]+' '+plot_data['model'][4], 'value':plot_data['path'][4]},
                        {'label': plot_data['metric'][5]+' '+plot_data['model'][5], 'value':plot_data['path'][5]},
                        {'label': plot_data['metric'][6]+' '+plot_data['model'][6], 'value':plot_data['path'][6]},
                        {'label': plot_data['metric'][7]+' '+plot_data['model'][7], 'value':plot_data['path'][7]},
                        {'label': plot_data['metric'][8]+' '+plot_data['model'][8], 'value':plot_data['path'][8]},
                    ],
                    value=plot_data['path'][0]),
                dcc.Graph(id='model_graph'),
                html.P(children = ['The figure below shows all three models and the three selected metrics. \
                                   From this figure we can see the HGBR model has the highest explained variance, and the \
                                   lowest RMSE and mean poisson deviance  ',
                                   

                ]),
                html.Img(src='https://raw.githubusercontent.com/CMonnin/Rail_Project/4e17c841be4c310ea468ab83bc26f58a94e4bfc4/Assets/model_comp.png',style={'max-width': '100%', 'height': 'auto', 'width': 'auto'}),
                html.H4('Neural_nets'),
                html.P(children = ['The next approach was to construct a neural net using Keras front-end and Tesorflow back-end workflow. \
                                   Searching the literature revealed that a feed-forward neural net was a good approach to \
                                   use[2]. Rectified linear activation unit (ReLU) were used as the activation functions for the hidden \
                                   layers. Linear activation was used on the output layer. The negative values then need to be converted \
                                   to zeros. The plots below display the loss function and the validation metric (rmse) over the 300 epochs. \
                                   It is worth noting that many combinations of loss and output activations were attempted. However the combination\
                                   reported gave the best results',
                ]),
                html.Img(src='https://raw.githubusercontent.com/CMonnin/Rail_Project/main/Assets/NN_plots.png',style={'max-width': '100%', 'height': 'auto', 'width': 'auto'}),
                html.P(children = ['The table below shows the results of the models with HGBR being the best to model the count data. ',
                ]),
                dash_table.DataTable(
                    data=model_results_comparison,
                    columns=columns,
                    style_table={'width': '50%'},
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'padding': '5px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                ),
                html.H3('Conclusion'),
                html.P(children = ['The GLM model selection went well. Further exploration of hyperparameter tuning would be a logical next step\
                                   It is also evident that the data is in the form a zero inflated poisson and in future steps should be treated as such. \
                                   A method knows as segment analysis is often performed on these types of issues however due to time constraints I was unable\
                                   to attempt this approach as it requires use of GIS[3]. \
                                   if more work was to be done on this dataset. The neural net proved difficult to properly evaluate and would require \
                                   more exploration. ',
                ]),

            ]),
            style={
            "text-align": "justify",
            "width": "6"}
        )
    ]),
    dbc.Row([
            html.Cite(dcc.Link('TSBC datasets https://www.tsb.gc.ca/eng/stats/rail/data-5.html', href='https://www.tsb.gc.ca/eng/stats/rail/data-5.html')),
            html.Cite(dcc.Link('Transport Canada grade inventories https://open.canada.ca/data/en/dataset/d0f54727-6c0b-4e5a-aa04-ea1463cf9f4c',href='https://open.canada.ca/data/en/dataset/d0f54727-6c0b-4e5a-aa04-ea1463cf9f4c')),
            html.Cite(' [1] English, G. W., and T. W. Moynihan. "Causes of accidents and mitigation strategies." Kingston, ON, Canada: TranSys Research (2007).'),
            html.Cite('[2] Montesinos‐Lopez, Osval A., et al. "Application of a Poisson deep neural network model for the prediction of count data in genome‐based prediction." The Plant Genome 14.3 (2021): e20118.'),
            html.Cite('[3] Chow, Tavia, et al. "A GIS approach to the development of a segment-level derailment prediction model." Accident Analysis & Prevention 151 (2021): 105897')
    ]),
])

# Define the callback
@app.callback(
    Output('model_graph', 'figure'),
    [Input('model_dropdown', 'value')]
)
def update_graph(value):
    # Call plotting function
    # data = df_plot.loc[1]['path']
    figure = plotter_from_file(value)
    return figure


@app.callback(
    Output('EDA-plot1', 'figure'),
    [Input('EDA-dropdown1', 'value')]
)
def update_box_plot(value):
    figure = make_subplots(rows=1, cols=2, subplot_titles=('Box Plot','KDE Plot'))

    figure.add_trace(px.box(EDA_data, y=value, points='outliers').data[0], row=1, col=1)
    # sadly the boxen plot wasn't working the same as sns.bloxen plot so i left it out
    # figure.add_trace(px.box(EDA_data, y=value, points='outliers', boxmode='overlay', title='Boxen Plot').data[0], row=1, col=2)
    figure.add_trace(px.histogram(EDA_data, x=value, histnorm='probability density', title='KDE Plot').data[0], row=1, col=2)
    figure.update_layout(showlegend=False, title='EDA Plots | All Features')
    return figure 




if __name__ == "__main__":
    app.run_server(debug=True)




