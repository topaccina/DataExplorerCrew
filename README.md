## App purpose
Dash app to perform operations on an input dataset on the basis of user textual input.<br>
Crew of AI Agents, triggered by the user query (natural language is input), runs under the hood to perform data filtering, trasformation to get the expected output.<br>
No other Dash filters or other control are used. Crew intermediate steps generate structured data. <br>
One text field and submit button are the only user access points. <br>
Both the input and the output dataset are used to populate dedicated Dash ag-grid tables.<br>
A short analysis summary (~150 Words) is generated and shown in a dedicated app section. <br>
A dedicated agents help to generate an insightful Plotly visualization showing the analysis results given the user query <br>
Both The output data in csv format and  the analysis summary in txt format have been exported during dedicated agents operations.

CrewAI framework adopted to implement the Crew, custom and langchain tools have been implemented/integrated.<br>
Under development-to join the [Charming Data Community](https://charming-data.circle.so/) Project initiative <br>




## VIDEO TO BE UPDATED!!!!!

## Main App features
1. Dash App design
2. Crew Definition
3. Custom tool development to implement agents tasks <br>

## AI features details
1. CrewAI framework to implement the crew
2. Langchain framework (Python AI packages) for crew tools implementation
3. Openai gpt-4o,gpt-4-turbo  for the crew tasks 

## Known Code Limitations, potential improvements and  other important notes
1. Still under decelopment to manage exceptions, improve tasks description to increase the output results quality<br>


## App structure

```bash
dash-app-structure

|-- .env
|-- .gitignore
|-- License
|-- README.md
|-- assets  
|-- data
|   |-- input dataset
|-- output
|   |-- generated output files
|-- utils
|   |-- support.py
|-- main_app.py
|-- requirements.txt

```

<br>

## Subfolders Details
### utils
code to retrieve the environment vars
### data
input dataset. Sample in csv format
### output
output files generated from the crew data processing
### python version
python311
