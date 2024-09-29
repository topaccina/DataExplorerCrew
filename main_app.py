# langchain packages
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# data manipulations packages
import pandas as pd
import re

import warnings

warnings.filterwarnings("ignore")

# crewai package
from crewai import Agent, Task, Crew, Process

# for dash app development
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px

# for env var retrieval and import custom local functions
import os
import sys

sys.path.insert(0, "./utils")
from utils import get_openai_api_key

##################### API KEYS #############################################
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"  # "gpt-4-turbo"
############################################################################
#################### Data Loading #########################################
df = pd.read_csv("./data/honey_US.csv")
tmp = df.copy()
# keep data subset for developing purpose
tmp = tmp[tmp.year > 2018]
tmp_columns = tmp.columns.to_list()
################################################################################
################ Tools ##########################################################

# create pandas agent to be used in custom pandas tool
pandas_agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tmp,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
)
# custom pandas tool
pandas_tool = Tool(
    name="pandas dataframe agent",
    func=pandas_agent.run,
    description="useful to analyze pandas dataframe",
)

dataStore = dcc.Store(
    id="data-store", data=pd.DataFrame(data={"placeHolder": ["1"]}).to_dict("records")
)


def get_fig_from_code(code, tmp_df):
    local_variables = {"tmp_df": tmp_df}
    exec(code, {}, local_variables)
    return local_variables["fig"]


def export_csv(df_out: dict):
    import pandas as pd

    print("this is the input")
    print(df_out)
    tmp_df = pd.DataFrame()
    if type(df_out) == dict:

        tmp_df = pd.DataFrame(
            dict([(key, pd.Series(value)) for key, value in df_out.items()])
        )
        print(tmp_df)
        print("dataframe avaialalbe")
        tmp_df.to_csv("./output/HoneyTest.csv", index=False)
        dataStore.data = tmp_df.to_dict("records")

    else:
        print("no dataframe avaialable")
        tmp_df = pd.DataFrame({"data": "no data"})
        tmp_df.to_csv("./output/HoneyTest.csv", index=False)


export_tool = Tool(
    name="create dataframe from dictionary and export to csv",
    description="receive a  python dictionary, turn into pandas dataframe and export  the pandas dataframe into a csv",
    func=export_csv,
)


###################### Agents #########################################################
analyst_agent = Agent(
    role="Data Analyst",
    goal="Analyze a Pandas Dataframe starting from questions coming from the consultant agent",
    backstory="you are expert data scientist with well knowledge of python and pandas package to manipulate dataframes",
    allow_delegation=False,
    verbose=True,
    memory=True,
)

export_agent = Agent(
    role="Data Exporter",
    goal="Export the pandas dataframe resulting from data analyst agent",
    backstory="you are python expert with great knowledge of pandas",
    allow_delegation=False,
    verbose=True,
    memory=True,
)

consultant_agent = Agent(
    role="Customer Consultant",
    goal="understanding the user requests and translating into a list of operations  to be performed on the dataset in pandas dataframe format",
    backstory="you are customer support consultant with well knowledge of the data set format and python pandas experience"
    "You are very skilled in understanding the user requests and translating into list of operation to be performed on the dataset in pandas dataframe format",
    allow_delegation=False,
    verbose=True,
    memory=True,
)

reporter_agent = Agent(
    role="Results report writer",
    goal="create a short detailed executive summary of the analysis results from the data analyst",
    backstory="you are a communication expert with great background in reporting to the management audience the results of data analysis from the data analysis",
    allow_delegation=False,
    verbose=True,
    memory=True,
)

vizs_expert_agent = Agent(
    role="Python Plotly visualization expert",
    goal="generate the code to design the most insightful plotly figure based on the results from the data analyst and the user request",
    backstory="you are Python Plotly expert. You have great experience in designing insightful plotly visualization from the results received from the analysis experts",
    allow_delegation=False,
    verbose=True,
    memory=True,
)
############################ Tasks ###############################################################
data_analysis_task = Task(
    description="Analyze the pandas dataframe and provide the results on the basis questions coming from the consultant agent, the results should includes only the relevant columns to asnwe to the {user_query}",
    expected_output="resulting pandas dataframe into a python dictionary with a key for each dataframe column and for values the lists of corresponding  columns values",
    tools=[pandas_tool],
    agent=analyst_agent,
)
export_task = Task(
    description="export in csv format the output python dictionary from the data analyis task",
    expected_output="csv format output file",
    tools=[export_tool],
    agent=export_agent,
    context=[data_analysis_task],
)


consultant_task = Task(
    description="understanding the the {user_query} and generating a clear description of the operation to be performed on the dataset in pandas dataframe format",
    expected_output="provide a clear question to the data analyst agent that need to operate on the dataframe with the following columns {tmp_columns}",
    tools=[],
    agent=consultant_agent,
)

reporter_task = Task(
    description=(
        "1. Use the data analisys results to craft a paragraph "
        "considering the analysis goal based on the {user_query}.\n"
        "2. highlight the main trends and numbers.\n"
        "3. Ensure that the summary is coincise with clear data analysis conclusion "
        "for a management audience. Max 150 words. "
    ),
    expected_output="A coincise paragraph "
    "summarizing the analysis results for management audience, "
    "starting by recalling the purpose of the analysis based on the {user_query},"
    "highlight the trends and main number",
    agent=reporter_agent,
    output_file="./output/summary_mgmt.txt",
)

vizs_task = Task(
    description=(
        "1. use the pandas dataframe stored in the local variable called tmp_df with the following list of columns {tmp_df_columns}"
        "2. the dataframe tmp_df is ready for the visualization. No other preprocessing is needed."
        "2. use plotly.express library for the visualization"
        "3. use only the {tmp_df_columns} variables from the tmp_df dataframe to create the code visualization"
        "3. assign the viszualization code in a variable called fig"
        "4. generate the code to get the most insightful plotly visualization  with title."
        "5. consider the user request {user_query} to define the title"
        "6.according with the visualization type define properties to improve readability, for example add markers or annotations or legend"
    ),
    expected_output=(
        "the code of the plotly express figure saved to fig variable, use the dataframe tmp_df with the the following list of columns {tmp_df_columns}   for the visualization"
        "use ONLY the the tmp_df columns exact name from the list {tmp_df_columns} to define the plotly visualization code"
        "the dataframe tmp_df is ready for the visualization. No other preprocessing is needed"
        "use the user request {user_query} to define the title"
    ),
    agent=vizs_expert_agent,
    tools=[],
)

# Create crew for data analysis and reporting
crew = Crew(
    agents=[consultant_agent, analyst_agent, export_agent, reporter_agent],
    tasks=[consultant_task, data_analysis_task, export_task, reporter_task],
    process=Process.sequential,
    verbose=True,
)

# create a crew for visualization - to be expanded with more agents and features
crew2 = Crew(
    agents=[vizs_expert_agent],
    tasks=[vizs_task],
    process=Process.sequential,
    verbose=True,
)

# dash app

app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

# ag-grid table to show the input dataset
columnDefs = [{"field": x} for x in tmp.columns]
grid = dag.AgGrid(
    id="get-started-example-basic",
    rowData=tmp.to_dict("records"),
    columnDefs=columnDefs,
)
# input field for user query
input_field = dbc.Input(id="user-input", placeholder="Type something...", type="text")
# submit the user query
submit_button = dbc.Button(
    "Click me", id="example-button", className="me-2", n_clicks=0
)

# main app layout
app.layout = dbc.Container(
    [
        dbc.Alert([html.H2("Data Explorer Crew")], color="success", className="m-2"),
        dbc.Container(
            [
                dbc.Row(
                    [dbc.Col([dbc.Label("Input Dataset"), html.Div([grid])])],
                    className="p-2 mt-2",
                ),
                dbc.Row(
                    [dbc.Col([dbc.Label("User Query"), html.Div([input_field])])],
                    className="p-1 mt-3 ",
                ),
                dbc.Row(
                    [dbc.Col([html.Div([submit_button])])],
                    className="p-2 mt-1 mb-2",
                ),
                # populated with the callback output. This is for the crews results
                dbc.Row(
                    [dbc.Col([dcc.Loading([], id="output")])],
                    className="p-2 mt-1",
                    align="center",
                ),
                # dcc store component to save intermediate data.
                dbc.Row(dbc.Col(dataStore)),
            ]
        ),
    ],
    className="p-5",
)


# main callback triggered by the user query. activate the crews and collect the output results showed in the dash app
@callback(
    Output("output", "children"),
    [Input("example-button", "n_clicks")],
    [State("user-input", "value")],
    prevent_initial_call=True,
)
def output_text(n, value):
    user_query = {
        "user_query": value,
        "tmp_columns": tmp_columns,
    }
    result = crew.kickoff(inputs=user_query)

    tmp_df = pd.DataFrame(dataStore.data)
    columnDefs = [{"field": x} for x in tmp_df.columns]
    table = dag.AgGrid(
        id="get-started-example-basic",
        rowData=tmp_df.to_dict("records"),
        columnDefs=columnDefs,
    )
    with open("./output/summary_mgmt.txt", "r") as file:
        data = file.read().rstrip()

    tmp_df_columns = tmp_df.columns.to_list()

    # for viz pupose
    user_query2 = {
        "user_query": value,
        "tmp_df_columns": tmp_df_columns,
        "summary": reporter_task.output.raw,
    }
    result2 = crew2.kickoff(inputs=user_query2)

    # parsing the code snippet for the plotly viz
    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", result2.raw, re.DOTALL)
    print("this is what I got now")
    viz = html.P("no results")
    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r"(?m)^\s*fig\.show\(\)\s*$", "", code_block)
        fig = get_fig_from_code(cleaned_code, tmp_df)
        viz = dcc.Graph(figure=fig)

    # crew output container
    out_content = (
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            [
                                                table,
                                            ],
                                            label="Output Dataset",
                                            className="m-3",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.P(data),
                                            ],
                                            label="Analysis Summary",
                                            className="m-3",
                                        ),
                                        dbc.Tab(
                                            [html.Div([viz], className="m-2")],
                                            label="Plot",
                                            className="m-3",
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
            ]
        ),
    )

    return out_content


if __name__ == "__main__":
    app.run_server(debug=True)
