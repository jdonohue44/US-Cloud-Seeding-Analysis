import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import geopandas as gpd
import plotly.express as px

# column_names = ['noaa_pdf','file_size(KB)','start_date',
#                 'end_date','season','target_area', 'year', 
#                 'state', 'type_of_agent', 'type_of_apparatus',
#                 'purpose','description','control_area',
#                 'total_amount_of_agent_used', 'total_mod_days'] 
# df = pd.read_csv('../results/FINAL.csv', names=column_names)

df = pd.read_csv('../data/FINAL.csv')

def standardize_state(state):
    state_mapping = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 
        'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID', 
        'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS', 
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 
        'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 
        'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC', 
        'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 
        'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 
        'wisconsin': 'WI', 'wyoming': 'WY', 'puerto rico': 'PR'
    }

    if not isinstance(state, str):
        return "UNKNOWN"
    state = re.sub(r'\s*\(.*?\)', '', state).strip().lower()
    return state_mapping.get(state, state.upper() if len(state) == 2 else "UNKNOWN")

def standardize_agent(agent):
    if pd.isna(agent):
        return 'Unknown'
    agent = agent.lower()
    if 'silver iodide' in agent:
        return 'Silver Iodide'
    elif 'sodium' in agent:
        return 'Sodium Compounds'
    elif 'carbon dioxide' in agent:
        return 'Carbon Dioxide'
    elif 'urea' in agent:
        return 'Urea'
    else:
        return 'Other'

###----- APPLY DATA STANDARDIZATION -----###
df['state'] = df['state'].apply(standardize_state)
df['type_of_agent'] = df['type_of_agent'].apply(standardize_agent)
df['year'] = pd.to_numeric(df['year'], errors='coerce')

###----- ANALYSIS AND VISUALIZATIONS -----###
# Where is WM being used in the US?
plt.figure(figsize=(12, 6))
state_counts = df['state'].value_counts()
sns.barplot(x=state_counts.index, y=state_counts.values)
plt.title('Weather Modification Usage by State')
plt.xlabel('State')
plt.ylabel('Number of WM Activities')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../plots/usage_by_state_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Is the use of WM increasing or decreasing over time?
plt.figure(figsize=(12, 6))
year_counts = df['year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title('Weather Modification Activities Over Time')
plt.xlabel('Year')
plt.ylabel('Number of WM Activities')
plt.tight_layout()
plt.savefig('../plots/time_series_line_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# What is WM typically used for?
purpose_counts = Counter(' '.join(df['purpose'].dropna()).lower().split())
common_words = ['and', 'the', 'to', 'for', 'of', 'in']
purpose_counts = {k: v for k, v in purpose_counts.items() if k not in common_words}
plt.figure(figsize=(12, 6))
purpose_df = pd.DataFrame.from_dict(purpose_counts, orient='index', columns=['count']).sort_values('count', ascending=False).head(10)
sns.barplot(x=purpose_df.index, y=purpose_df['count'])
plt.title('Most Common Words in WM Purpose Descriptions')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../plots/purpose_breakdown_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Categorize the purposes - NEW
# categorized_df, category_counts, multi_purpose_percentage = categorize_weather_modification_purposes(df)

# What types of materials are used?
plt.figure(figsize=(10, 6))
agent_counts = df['type_of_agent'].value_counts()
colors = sns.color_palette('pastel')[0:len(agent_counts)]
plt.pie(agent_counts.values, labels=agent_counts.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Types of Agents Used in Weather Modification')
plt.tight_layout()
plt.savefig('../plots/agent_type_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()

###----- MULTI-VARIABLE PLOTS -----###
# year, state, type_of_agent, type_of_apparatus, purpose, season
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='year', hue='state')
plt.title('Count of X Variable by Hue Variable')
plt.xlabel('X Variable')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../plots/state_and_year_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

df_grouped = df.groupby(['year', 'type_of_agent']).size().unstack()
df_grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Count of Year by Agent Type')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../plots/agent_and_year_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

###----- GEOSPATIAL PLOTS -----###
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
fig = px.choropleth(
    state_counts,
    locations='state',
    locationmode="USA-states",
    color='count',
    scope="usa",
    color_continuous_scale="Viridis",
    hover_name='state',
    hover_data=['count'],
    labels={'count': 'Row Count'}
)
fig.update_layout(
    title_text='Count of Rows by State',
    geo_scope='usa'
)
fig.write_image("../plots/usage_map.png")
# fig.show()