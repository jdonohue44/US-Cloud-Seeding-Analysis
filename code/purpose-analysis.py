import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

df = pd.read_csv('../data/FINAL.csv')

def categorize_weather_modification_purposes(df, purpose_column='purpose', plot=True, top_n=10):
    """
    Analyzes and categorizes the purposes of weather modification activities
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing weather modification data
    purpose_column : str
        Column name containing purpose descriptions
    plot : bool
        Whether to generate visualization plots
    top_n : int
        Number of top categories to display in plots
        
    Returns:
    --------
    tuple
        (categorized_df, category_counts, multi_purpose_percentage)
    """
    # Define categories and their related keywords
    categories = {
        'Snowpack Augmentation': ['snowpack', 'snow augment', 'enhance snowpack', 'increase snow', 
                                  'snowfall', 'snow pack', 'snow enhancement'],
        'Precipitation Enhancement': ['precipitation enhancement', 'precipitation augment', 
                                     'increase precipitation', 'rain enhancement', 'rainfall increase', 
                                     'enhance precipitation', 'augment precipitation', 'rain optimization'],
        'Hail Suppression': ['hail suppression', 'mitigate hail', 'alleviate hail', 'hail damage'],
        'Fog Suppression': ['fog suppression', 'fog dissipation', 'alleviate fog', 'fog clearing'],
        'Water Supply Management': ['water supply', 'increase inflow', 'runoff', 'irrigation', 
                                   'drainage district', 'basin', 'reservoir', 'aquifer', 'recharge'],
        'Research & Development': ['research', 'study', 'verify', 'evaluate', 'assess'],
        'Airport Operations': ['airport', 'landing', 'takeoff'],
        'Ski Area Enhancement': ['ski area', 'ski resort'],
        'Fire Management': ['wildfire', 'firefighter'],
        'Temperature Control': ['temperature', 'global temperature']
    }
    
    def categorize_purpose(purpose):
        if pd.isna(purpose):
            return ['Undetermined']
        
        purpose = str(purpose).lower()
        matched_categories = []
        
        # Check for each category
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in purpose:
                    matched_categories.append(category)
                    break  # Move to next category after finding a match
        
        # Handle cases with no matches
        if not matched_categories:
            if 'weather modification' in purpose or 'cloud seeding' in purpose:
                matched_categories.append('General Weather Modification')
            elif 'undetermined' in purpose or 'not explicitly stated' in purpose:
                matched_categories.append('Undetermined')
            else:
                matched_categories.append('Other')
        
        return matched_categories
    
    # Apply categorization
    categorized_purposes = df[purpose_column].fillna('Undetermined').apply(categorize_purpose)
    
    # Count category occurrences
    all_categories = []
    for categories_list in categorized_purposes:
        all_categories.extend(categories_list)
    
    category_counts = Counter(all_categories)
    
    # Calculate multi-purpose percentage
    multi_purpose_count = sum(1 for cats in categorized_purposes if len(cats) > 1)
    multi_purpose_percentage = (multi_purpose_count / len(df)) * 100
    
    # Generate a new column with primary category (first in the list)
    df_with_category = df.copy()
    df_with_category['primary_purpose_category'] = categorized_purposes.apply(lambda x: x[0] if x else 'Undetermined')
    df_with_category['all_purpose_categories'] = categorized_purposes.apply(lambda x: ', '.join(x))
    df_with_category['has_multiple_purposes'] = categorized_purposes.apply(lambda x: len(x) > 1)
    
    # Generate plots if requested
    if plot:
        # Plot 1: Category distribution
        plt.figure(figsize=(14, 8))
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        categories, counts = zip(*sorted_counts[:top_n])
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=list(counts), y=list(categories))
        plt.title(f'Top {top_n} Weather Modification Purpose Categories')
        plt.xlabel('Count')
        plt.tight_layout()
        
        # Plot 2: Multi-purpose proportion
        plt.subplot(1, 2, 2)
        multi_purpose_data = [multi_purpose_count, len(df) - multi_purpose_count]
        plt.pie(multi_purpose_data, 
                labels=['Multi-purpose', 'Single purpose'], 
                autopct='%1.1f%%',
                colors=['#ff9999','#66b3ff'])
        plt.title('Proportion of Multi-purpose Weather Modification Activities')
        plt.tight_layout()
        
        plt.show()
    
    return df_with_category, category_counts, multi_purpose_percentage

def analyze_regional_purposes(df, region_column, purpose_category_column='primary_purpose_category'):
    """
    Analyzes how purpose categories vary by region
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with categorized purposes
    region_column : str
        Column name containing region information
    purpose_category_column : str
        Column name containing purpose categories
        
    Returns:
    --------
    pandas.DataFrame
        Cross-tabulation of purposes by region
    """
    if region_column not in df.columns:
        print(f"Warning: {region_column} column not found in DataFrame")
        return None
    
    # Create a cross-tabulation
    crosstab = pd.crosstab(df[region_column], df[purpose_category_column])
    
    # Normalize to get percentages
    normalized_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # Visualize
    plt.figure(figsize=(16, 10))
    sns.heatmap(normalized_crosstab, annot=True, cmap="YlGnBu", fmt='.1f')
    plt.title('Weather Modification Purposes by Region (% within region)')
    plt.tight_layout()
    plt.show()
    
    return crosstab, normalized_crosstab

# RUN: Categorize the purposes
categorized_df, category_counts, multi_purpose_percentage = categorize_weather_modification_purposes(df)
