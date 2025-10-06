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
import kaleido
from PIL import Image
import io

# Load Data
df = pd.read_csv('../data/cleaned_cloud_seeding_us_2000_2025.csv')

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

def rgb_string_to_tuple(rgb_str):
    rgb_values = rgb_str.strip('rgb()').split(',')
    return tuple(int(v)/255 for v in rgb_values)

# Set overall styles for plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

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
    state_counts[state_counts['count'] > 0],
    locations='state',
    locationmode="USA-states",
    color='count',
    scope="usa",
    color_continuous_scale=px.colors.sequential.Blues[2:], 
    hover_name='state',
    hover_data=['count']
)
fig.update_layout(geo_scope='usa')

# Save with proper 300 DPI metadata
img_bytes = fig.to_image(format="png", width=1920, height=1080, scale=5)
img = Image.open(io.BytesIO(img_bytes))

# Convert RGBA to RGB (remove transparency)
if img.mode == 'RGBA':
    # Create a white background
    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
    img = rgb_img

img.save("/Users/jareddonohue/Desktop/weather_modification_map.jpg", "JPEG", dpi=(300, 300), quality=95)
# Show Plot
# fig.show()

# 2) Stacked Bars - Number of Weather Modification Activities Per Year
states_with_counts = clean_df['state'].value_counts().index
year_state_counts = clean_df[clean_df['state'].isin(states_with_counts)]
grouped = year_state_counts.groupby(['year', 'state']).size().unstack(fill_value=0)
grouped = grouped.sort_index()
n_states = len(grouped.columns)
samplepoints = [i / (n_states - 1) * 0.6 + 0.4 for i in range(n_states)]  # from 0.4 to 1.0 scaled to n_states
custom_blues = pc.sample_colorscale(pc.sequential.Blues, samplepoints)[::-1]
colors = [rgb_string_to_tuple(c) for c in custom_blues]
plt.figure(figsize=(10,6))
# y-grid only
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
# plt.figure(figsize=(14, 7))
bottom = [0] * len(grouped)
for i, state in enumerate(grouped.columns):
    plt.bar(grouped.index, grouped[state], bottom=bottom, label=state, color=colors[i])
    bottom = [sum(x) for x in zip(bottom, grouped[state])]
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Weather Modification Activities", fontsize=14)
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save as 300 DPI JPG
plt.savefig("/Users/jareddonohue/Desktop/cloud_seeding_timeline_stacked_bars.jpg", dpi=300, format="jpg", bbox_inches="tight")
# Show Plot
# plt.show()

# 3) Horizontal Bars - Breakdown of purposes
purpose_series = df["purpose"].dropna()
split_purposes = purpose_series.apply(lambda s: [p.strip() for p in s.split(",") if p.strip()])
flat_grouped = [item for sublist in split_purposes for item in sublist]
grouped_counts = Counter(flat_grouped)
grouped_df = pd.DataFrame(grouped_counts.items(), columns=["purpose", "count"]).sort_values(by="count", ascending=False)
top_grouped_df = grouped_df.head(10)
n = len(top_grouped_df)
cmap = get_cmap('Blues')
colors = [cmap(0.4 + 0.6 * (i / (n - 1))) for i in range(n)][::-1]
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    y=top_grouped_df["purpose"],
    x=top_grouped_df["count"],
    palette=colors
)
# ax.bar_label(ax.containers[0], fontsize=12, padding=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.6)
# ax.bar_label(ax.containers[0], fontsize=12, padding=3, color="black", label_type="edge")
sns.barplot(
    y=top_grouped_df["purpose"],
    x=top_grouped_df["count"],
    palette=colors
)
plt.xlabel("Number of Activities", fontsize=14)
plt.ylabel("Purpose", fontsize=14)
plt.tight_layout()

# Save as 300 DPI JPG
plt.savefig("/Users/jareddonohue/Desktop/cloud_seeding_purpose_horizontal_bars.jpg", dpi=300, format="jpg", bbox_inches="tight")
# Show Plot
# plt.show()

# 4) Heatmap - Breakdown of agent and apparatus
df = df[df['agent'].str.len() <= 50]
df = df[df['agent'].notna() & df['apparatus'].notna()]
agg = df.groupby(['agent', 'apparatus']).size().reset_index(name='count')
pivot_table = agg.pivot(index='agent', columns='apparatus', values='count').fillna(0)
# Remove rows where the sum across apparatus is only 1
pivot_table = pivot_table[pivot_table.sum(axis=1) > 1]
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
    linewidths=0.5,
    linecolor='white',
    xticklabels=True,
    yticklabels=True
)
plt.xlabel("Apparatus", fontsize=14, labelpad=12)
plt.ylabel("Agent", fontsize=14, labelpad=12)
plt.xticks(fontsize=12, rotation=30, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()

# Save as 300 DPI JPG
plt.savefig("/Users/jareddonohue/Desktop/cloud_seeding_agent_apparatus.jpg", dpi=300, format="jpg", bbox_inches="tight")
# Show Plot
# plt.show()
