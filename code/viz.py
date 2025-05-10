from collections import Counter
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import get_cmap
import seaborn as sns
import plotly.express as px
import plotly.colors as pc

# Load Data
df = pd.read_csv('../data/cloud_seeding_us_2000_2025.csv')

# State name to abbreviation mapping
state_abbrev = {
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
    'wisconsin': 'WI', 'wyoming': 'WY'
}

# Updated mapping with broader normalization
purpose_mapping = {
    "increase rain": ["rain enhancement", "rainfall enhancement", "rainfall increase", "increase rainfall", "rain augmentation", "augment winter rainfall"],
    "increase snowpack": [
        "augment snowpack", "snowpack augmentation", "snow pack augmentation", "increase snowpack",
        "augment winter snowpack", "increase early season snowpack",
        "augment mountain snowpack", "increase high-elevation snowpack",
        "augment early season snowpack", "snowpack increase", "increase high elevation snowpack"
    ],
    "increase precipitation": [
        "precipitation augmentation", "augment precipitation",
        "increase precipitation", "precipitation increase",
        "snow precipitation augmentation", "precipitation enhancement",
        "augment winter precipitation"
    ],
    "increase snowfall": [
        "snowfall augmentation", "increase snowfall", "snow augmentation"
    ],
    "hail suppression": [
        "hail suppression", "hail damage mitigation", "hailfall damage mitigation"
    ],
    "fog suppression": [
        "fog dissipation", "fog suppression", "fog dispersal", "fog clearing"
    ],
    "increase runoff": [
        "increase runoff", "increase dry season runoff", "increased dry season runoff", "increase dry-season runoff", "augment runoff",
        "increase subsequent runoff", "increase inflow to reservoir",
        "increase inflow to twitchell reservoir", "increase inflow to great salt lake"
    ],
    "increase water supply": ["increase water supply", "augment water supply"],
    "research": ["research", "research and development", "study"],
    "drought relief": ["drought relief"],
    "assist firefighting": ["assist firefighting"],
}

def normalize_agent(agent):
    if not isinstance(agent, str):
        return agent
    agent = agent.lower().strip()

    # Drop unwanted entries
    if 'sea salt' in agent or 'liquid water' in agent or 'water droplets' in agent or 'condensation freezing seeding agent' in agent:
        return None

    # Normalize chemical shorthand
    agent = agent.replace('ammonia', 'ammonium') 
    agent = agent.replace('agi', 'silver iodide')
    agent = agent.replace('nh4i', 'ammonium iodide')
    agent = agent.replace('nh41', 'ammonium iodide')
    agent = agent.replace('-', ' ')
    agent = agent.replace('ammonia iodide', 'ammonium iodide') 
    agent = agent.replace('silver iodide ammonium iodide', 'silver iodide, ammonium iodide')
    agent = agent.replace('kcl', 'potassium chloride')

    # Consolidate hygroscopic variations
    if 'hygroscopic' in agent:
        return 'silver iodide, hygroscopic'

    # Consolidate other complex known mixes
    if 'acetone' in agent:
        return 'silver iodide, acetone'
    if 'dry ice' in agent or 'carbon dioxide' in agent:
        return 'silver iodide, dry ice'
    if 'silver iodate' in agent:
        return 'silver iodide'
    
    if 'silver iodide' in agent and 'ammonium iodide' in agent:
        return 'silver iodide, ammonium iodide'
    if 'ammonium iodide' in agent:
        return 'ammonium iodide'


    return agent.strip()

def rgb_string_to_tuple(rgb_str):
    rgb_values = rgb_str.strip('rgb()').split(',')
    return tuple(int(v)/255 for v in rgb_values)

def normalize_purpose(purpose):
    if pd.isna(purpose):
        return []
    purpose = purpose.lower()
    parts = re.split(r'[,;/&]+', purpose)
    parts = [p.strip() for p in parts if p.strip()]
    grouped = set()
    for part in parts:
        matched = False
        for canonical, variants in purpose_mapping.items():
            if any(v in part for v in variants):
                grouped.add(canonical)
                matched = True
                break
        if not matched:
            grouped.add(part)  # keep raw if no match
    return list(grouped)

# 1) Map - Number of Weather Modification Activities Per State
expanded_rows = []
for _, row in df.iterrows():
    states = [s.strip().lower() for s in str(row['state']).split(',')]
    for state in states:
        abbrev = state_abbrev.get(state)
        if abbrev:
            new_row = row.copy()
            new_row['state'] = abbrev
            expanded_rows.append(new_row)
clean_df = pd.DataFrame(expanded_rows)
state_counts = clean_df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
fig = px.choropleth(
    state_counts[state_counts['count'] > 2],
    locations='state',
    locationmode="USA-states",
    color='count',
    scope="usa",
    color_continuous_scale=px.colors.sequential.Blues[2:], 
    hover_name='state',
    hover_data=['count']
)
fig.update_layout(geo_scope='usa')
fig.show()

# 2) Stacked Bars - Number of Weather Modification Activities Per Year
top_states = clean_df['state'].value_counts().head(9).index
year_state_counts = clean_df[clean_df['state'].isin(top_states)]
grouped = year_state_counts.groupby(['year', 'state']).size().unstack(fill_value=0)
grouped = grouped.sort_index()

# Generate 9 reversed custom blue shades (skip lightest, start at 0.3)
samplepoints = [i / 8 * 0.6 + 0.4 for i in range(9)]  # from 0.4 to 1.0
custom_blues = pc.sample_colorscale(pc.sequential.Blues, samplepoints)[::-1]  # reverse

colors = [rgb_string_to_tuple(c) for c in custom_blues]

# Plot
plt.figure(figsize=(14, 7))
bottom = [0] * len(grouped)
for i, state in enumerate(grouped.columns):
    plt.bar(grouped.index, grouped[state], bottom=bottom, label=state, color=colors[i])
    bottom = [sum(x) for x in zip(bottom, grouped[state])]

# plt.title("Weather Modification Activities Per Year")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Weather Modification Activities", fontsize=14)
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3) Horizontal Bars - Breakdown of purposes
purpose_series = df["purpose"].dropna()
all_grouped = purpose_series.apply(normalize_purpose)
flat_grouped = [item for sublist in all_grouped for item in sublist]
grouped_counts = Counter(flat_grouped)
grouped_df = pd.DataFrame(grouped_counts.items(), columns=["purpose", "count"]).sort_values(by="count", ascending=False)
top_grouped_df = grouped_df.head(10)
n = len(top_grouped_df)
cmap = get_cmap('Blues')
colors = [cmap(0.4 + 0.6 * (i / (n - 1))) for i in range(n)][::-1]
plt.figure(figsize=(10, 6))
sns.barplot(
    y=top_grouped_df["purpose"],
    x=top_grouped_df["count"],
    palette=colors
)
# plt.title("Purpose of Cloud Seeding Activities (2000–2025)")
plt.xlabel("Number of Activities", fontsize=14)
plt.ylabel("Purpose", fontsize=14)
plt.tight_layout()
plt.show()

# 4) Heatmap - Breakdown of agent and apparatus
df['apparatus'] = df['apparatus'].str.strip().str.lower()
df['agent'] = df['agent'].str.strip().str.lower()
df['agent_group'] = df['agent'].apply(normalize_agent)
df = df[df['agent_group'].notna()]
df = df[df['agent'].str.len() <= 50]
# print(df['agent_group'].value_counts())
agg = df.groupby(['agent_group', 'apparatus']).size().reset_index(name='count')
pivot_table = agg.pivot(index='agent_group', columns='apparatus', values='count').fillna(0)
plotly_colors = px.colors.sequential.Blues[2:]
converted_colors = [rgb_string_to_tuple(c) for c in plotly_colors]
custom_blues = LinearSegmentedColormap.from_list("custom_blues", converted_colors)
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".0f",
    cmap=custom_blues,
    cbar_kws={'label': 'Number of Activities'},
    annot_kws={"size": 10},
    linewidths=0.25,
    linecolor='gray',
    xticklabels=True,
    yticklabels=True
)
# plt.title("Cloud Seeding Activities by Agent and Apparatus (2000–2025)", fontsize=16)
plt.xlabel("Apparatus", fontsize=14, labelpad=12)
plt.ylabel("Agent Group", fontsize=14, labelpad=12)
plt.xticks(fontsize=12, rotation=30, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
