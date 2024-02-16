from ast import literal_eval
from typing import Optional

import click
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px


class EraseOutliers():
    """
    Class-application that displays given dataset in
    scatterplot format. It allows to erase points on plot
    using plotly's lasso instrument.

    Such mechanics are usefull in case of extraordinary
    outliers that could be processed only by Machine Learning 
    or by hand

    There are some dropdowns and buttons in the app

    - `X-axis column` is the dropdown for column used as X-axis
    - `Y-axis column` is the dropdown for column used as Y-axis
    - `Erase` button is trigger to erase selected by lasso points
    - `Undo` button allows to return erased data on the previous step
    - `Path to save dataframe` is path to save the processed dataset
    - `Save extension` is extension for saved dataset. One of [`csv`, `parquet`, `json`, `pickle`]
    - `Save` button is trigger to save processed dataset

    """
    
    def __init__(self):
        self._app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        self._erased_data = []
        self._df = None
        self._x_col = ""
        self._y_col = ""
        
        self._load_oper = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "json": pd.read_json,
            "pickle": pd.read_pickle
        }
        
        self._save_oper = {
            "csv": lambda x: x.to_csv,
            "parquet": lambda x: x.to_parquet,
            "json": lambda x: x.to_json,
            "pickle": lambda x: x.to_pickle
        }
        
        # This is the way to share `self` with `callback` functions
        self._register_callbacks(self._app)
        
        
    def _register_callbacks(self, app):
           
        @app.callback(
        Output('indicator-graphic', 'figure'),
        Input('xaxis-column', 'value'),
        Input('yaxis-column', 'value'),
        )
        def update_graph(xaxis_column_name: str, yaxis_column_name: str):
            """Redraws plot with given columns

            :param xaxis_column_name: Column used as X-axis
            :type xaxis_column_name: str

            :param yaxis_column_name: Column used as Y-axis
            :type yaxis_column_name: str

            :return: Resulting plotly figure
            :rtype: plotly.graph_objects._figure.Figure
            """
            # All the `self` variables are shared between callbacks
            self._x_col = xaxis_column_name
            self._y_col = yaxis_column_name
            fig = px.scatter(data_frame=self._df, x=xaxis_column_name, y=yaxis_column_name)
            # Add behavior for figure
            fig.update_layout(clickmode='event+select')
            return fig
            
            
        @app.callback(
            Output('indicator-graphic', 'figure', allow_duplicate=True),
            Input('submit-button-state', 'n_clicks'),
            State('indicator-graphic', 'selectedData'),    
            prevent_initial_call=True
        )
        def erase_selected_data(
            n_clicks: int,
            selectedData: dict
        ):
            """Method of button to erase selected by plotly lasso or rectangle points

            :param n_clicks: just arg from button
            :type n_clicks: int
            :param selectedData: Dict with data from lasso/rectangle operation
            :type selectedData: dict
            :raises ValueError: If lasso caught no points
            :return: Resulting plotly figure
            :rtype: plotly.graph_objects._figure.Figure
            """
            # Get selected points from lasso
            points_list = selectedData["points"]
            
            if not points_list:
                raise ValueError("No points are selected")
                        
            points_flatten_list = {k: [dic[k] for dic in points_list] for k in points_list[0]}
            
            points_df = pd.DataFrame(
                {self._x_col: points_flatten_list["x"], self._y_col: points_flatten_list["y"]}
            ).drop_duplicates()
            
            print(f"Request processing. Got {len(points_list)} points, filtering {len(points_df)} points")
            
            temp = self._df.copy()
            temp["index"] = temp.index
            res = temp.merge(points_df, how='inner', on=[self._x_col,self._y_col])
            
            res = res.set_index("index")
            self._erased_data.append(res)
            
            self._df.drop(res.index, inplace=True)
            
            fig = px.scatter(data_frame=self._df, x=self._x_col, y=self._y_col)
            fig.update_layout(clickmode='event+select')
            
            return fig
        

        @app.callback(
            Output("hidden-div", 'children'),
            Input('save-button-state', 'n_clicks'),
            State('target-path', 'value'),
            State('save-ext', 'value'),        
            prevent_initial_call=True
        )
        def save_current_df(
            n_clicks: int,
            target_path: str,
            target_ext: str
        ):
            """Method of button for saving DataFrame at current condition

            :param n_clicks: just button arg
            :type n_clicks: int
            :param target_path: Path for saved dataset
            :type target_path: str
            :param target_ext: Extension of saved dataset. One of [`csv`, `parquet`, `json`, `pickle`]
            :type target_ext: str
            """
            self._save_oper[target_ext](self._df)(target_path)
            
            
        @app.callback(
            Output("indicator-graphic", 'figure',  allow_duplicate=True),
            Input('undo-button-state', 'n_clicks'),  
            prevent_initial_call=True
        )
        def undo_erasing(
            n_clicks: int
        ):
            """Method of button to undo previous erasig operations.
            Erasing supports history

            :param n_clicks: just button arg
            :type n_clicks: int
            :raises ValueError: If erasing stack is empty
            :return: Resulting plotly figure
            :rtype: plotly.graph_objects._figure.Figure
            """
            if not self._erased_data:
                 raise ValueError("There are nowhere to undo")
            self._df = pd.concat([self._df, self._erased_data.pop().copy()]).sort_index()
            fig = px.scatter(data_frame=self._df, x=self._x_col, y=self._y_col)
            fig.update_layout(clickmode='event+select')
            
            return fig
        

    def read_by_path(
        self,
        source_file_path: str,
        source_extension: str,
        sorce_file_kwargs: dict = {}, 
    ):
        """Method to load DataFrame to the class object by reading a file

        :param source_file_path: Path to source file
        :type source_file_path: str
        :param source_extension: Source file extension. One of [`csv`, `parquet`, `json`, `pickle`]
        :type source_extension: str
        :param sorce_file_kwargs: Kwargs for pandas reading function, defaults to {}
        :type sorce_file_kwargs: dict, optional
        :raises ValueError: if wrong extension is passed
        """
        if source_extension not in self._load_oper:
            raise ValueError(
                f"File extension {source_extension} is not supportable. Try one of {list(self._load_oper.keys())}"
            )
        self._df = self._load_oper[source_extension](source_file_path, **sorce_file_kwargs)
        self._erased_data = []
    
        
    def read_by_object(
        self,
        df: pd.DataFrame
    ):
        """Method to pass DataFrame object to the class object

        :param df: Source dataset
        :type df: pd.DataFrame
        """
        self._df = df.copy()
        self._erased_data = []
    
        
    def start_plot(self):
        """Here the construction begins

        :raises ValueError: If no dataset was passed
        :raises ValueError: If dataset is too short
        :raises ValueError: If there are not enough columns in dataset
        """
        if self._df is None:
            raise ValueError("Inner DataFrame has to be set via `read_by_path` or `read_by_object` funcs")
        
        if len(self._df) < 2:
            raise ValueError("Inner DataFrame is too short. Should have at least 2 rows")
        
        if len(self._df.columns) < 2:
            raise ValueError("Inner DataFrame should have at least 2 columns")
            
        x_axis_dropdown = html.Div([
            dbc.Label("X-axis column", html_for="xaxis-column"),
            dcc.Dropdown(
                self._df.columns,
                self._df.columns[0],
                id='xaxis-column'
            )
        ], className="mt-2")
        
        y_axis_dropdown = html.Div([
            dbc.Label("Y-axis column", html_for="yaxis-column"),
            dcc.Dropdown(
                self._df.columns,
                self._df.columns[1],
                id='yaxis-column'
            )
        ], className="mt-2")
        
        submit_undo_buttons = html.Div(
            [
                html.Div([
                    dbc.Button(id='submit-button-state', n_clicks=0, children='Erase', color="primary"),
                ], className="mt-2", style= {'width': '48%',  'display': 'inline-block'}),
                html.Div([
                    dbc.Button(id='undo-button-state', n_clicks=0, children='Undo', color="secondary"),
                ], className="mt-2", style= {'width': '48%', "float":"right", 'display': 'inline-block'}),
            ], className="mt-2"
        )
        
        save_input = html.Div([
            dbc.Label("Path to save dataframe", html_for="target-path"),
            dcc.Input(
                id='target-path',
                type='text',
                value="test.parquet"
            ),
        ], className="mt-2")
        
        save_extension = html.Div([
            dbc.Label("Save extension", html_for="save-ext"),
            dcc.Dropdown(
                ["csv", "parquet", "json", "pickle"],
                "parquet",
                id='save-ext'
            ),
        ], className="mt-2")
            
        save_button = html.Div(
            [
                dbc.Button(id='save-button-state', n_clicks=0, children='Save', size="lg", color="success"),
            ],
            className="mt-2"
        )
        
        control_panel = dbc.Card(
            dbc.CardBody(
                [x_axis_dropdown, y_axis_dropdown, submit_undo_buttons, save_input, save_extension, save_button],
                className="bg-light",
            )
        )
        
        graph = dbc.Card(
            [html.Div(id="error_msg", className="text-danger"), dcc.Graph(id="indicator-graphic")]
        )
        
        self._app.layout = html.Div([
            dbc.Row([dbc.Col(control_panel, md=4), dbc.Col(graph, md=8)]),
            html.Div(id="hidden-div")
            ]
        )
        
        self._app.run(debug=True, host="0.0.0.0", port=8050)



@click.option("--source_file_path", type=str, )
@click.option("--source_file_extension", type=str)
@click.option("--source_file_kwargs", type=str)
def func_erase_outliers(
    source_file_path: Optional[str] = None,
    source_file_extension: Optional[str] = None,
    source_file_kwargs: Optional[str] = None
):
    eo = EraseOutliers()

    if all((arg is None for arg in [source_file_path, source_file_extension, source_file_kwargs])):
        df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
        eo.read_by_object(df)
    elif source_file_path is not None and source_file_extension is not None:
        source_file_kwargs = {} if source_file_kwargs is None else literal_eval(source_file_kwargs)
        eo.read_by_object(source_file_path, source_file_extension, **source_file_kwargs)
    else:
        raise ValueError("One shoud either pass file path, file extension and pandas.read kwargs or pass nothing")
    
    eo.start_plot()


f_erase_outliers = click.command()(func_erase_outliers)


if __name__ == "__main__":
    f_erase_outliers()
