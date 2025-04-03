import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('poster_presentation_data.csv')

# Load the dataset
df = load_data()

class Dashboard:
    def __init__(self, df):
        self.df = df
        
    def display(self):
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Day filter
        selected_days = st.sidebar.multiselect(
            "Select Days",
            options=sorted(self.df['Day'].unique()),
            default=sorted(self.df['Day'].unique())
        )
        
        # Track filter
        selected_tracks = st.sidebar.multiselect(
            "Select Tracks",
            options=sorted(self.df['Track'].unique()),
            default=sorted(self.df['Track'].unique())
        )
        
        # College filter
        selected_colleges = st.sidebar.multiselect(
            "Select Colleges",
            options=sorted(self.df['College'].unique()),
            default=sorted(self.df['College'].unique())
        )
        
        # Apply filters
        filtered_df = self.df[
            (self.df['Day'].isin(selected_days)) &
            (self.df['Track'].isin(selected_tracks)) &
            (self.df['College'].isin(selected_colleges))
        ]
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Participants",
                len(filtered_df),
                f"{len(filtered_df) - len(self.df)} from filtered view"
            )
        
        with col2:
            st.metric(
                "Unique Colleges",
                filtered_df['College'].nunique(),
                f"{filtered_df['College'].nunique() - self.df['College'].nunique()} from filtered view"
            )
        
        # Visualizations
        st.subheader("Participation Trends")
        
        # Day-wise participation
        day_counts = filtered_df['Day'].value_counts().sort_index()
        fig_day = px.bar(
            x=day_counts.index,
            y=day_counts.values,
            title="Participants by Day",
            labels={'x': 'Day', 'y': 'Number of Participants'}
        )
        st.plotly_chart(fig_day, use_container_width=True)
        
        # Track distribution
        track_counts = filtered_df['Track'].value_counts()
        fig_track = px.pie(
            values=track_counts.values,
            names=track_counts.index,
            title="Distribution across Tracks"
        )
        st.plotly_chart(fig_track, use_container_width=True)
        
        # College participation
        college_counts = filtered_df['College'].value_counts().head(10)
        fig_college = px.bar(
            x=college_counts.values,
            y=college_counts.index,
            orientation='h',
            title="Top 10 Participating Colleges",
            labels={'x': 'Number of Participants', 'y': 'College'}
        )
        st.plotly_chart(fig_college, use_container_width=True)
        
        # Additional insights
        st.subheader("Key Insights")
        
        # Most common tracks
        track_insights = filtered_df['Track'].value_counts().head(3)
        st.write("Top 3 Most Popular Tracks:")
        for track, count in track_insights.items():
            st.write(f"- {track}: {count} participants")
            
        # Day-wise participation insights
        day_insights = filtered_df['Day'].value_counts().sort_index()
        st.write("\nDay-wise Participation:")
        for day, count in day_insights.items():
            st.write(f"- Day {day}: {count} participants")

# Main dashboard title
st.title("National Poster Presentation Dashboard")

# Create two columns for the first row of charts
col1, col2 = st.columns(2)

# 1. Track-wise participation (Bar Chart)
with col1:
    track_counts = df['Track'].value_counts()
    fig_track = px.bar(
        x=track_counts.index,
        y=track_counts.values,
        title="Track-wise Participation",
        labels={'x': 'Track', 'y': 'Number of Participants'},
        color=track_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_track, use_container_width=True)

# 2. Day-wise participation trends (Line Graph)
with col2:
    day_counts = df['Day'].value_counts().sort_index()
    fig_day = px.line(
        x=day_counts.index,
        y=day_counts.values,
        title="Day-wise Participation Trends",
        labels={'x': 'Day', 'y': 'Number of Participants'},
        markers=True
    )
    st.plotly_chart(fig_day, use_container_width=True)

# Create two columns for the second row of charts
col3, col4 = st.columns(2)

# 3. College-wise distribution (Pie Chart)
with col3:
    college_counts = df['College'].value_counts()
    fig_college = px.pie(
        values=college_counts.values,
        names=college_counts.index,
        title="College-wise Distribution",
        hole=0.4
    )
    st.plotly_chart(fig_college, use_container_width=True)

# 4. State-wise participation (Choropleth Map)
with col4:
    state_counts = df['State'].value_counts()
    fig_map = px.choropleth(
        locations=state_counts.index,
        locationmode="USA-states",
        color=state_counts.values,
        scope="usa",
        title="State-wise Participation",
        color_continuous_scale="Viridis",
        labels={'color': 'Number of Participants'}
    )
    st.plotly_chart(fig_map, use_container_width=True)

# 5. Heatmap of participation density
st.subheader("Participation Density Heatmap")
pivot_table = pd.pivot_table(
    df,
    values='Participant_ID',
    index='Day',
    columns='Track',
    aggfunc='count',
    fill_value=0
)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=pivot_table.values,
    x=pivot_table.columns,
    y=pivot_table.index,
    colorscale='Viridis',
    text=pivot_table.values,
    texttemplate='%{text}',
    textfont={"size": 12},
    hoverongaps=False
))

fig_heatmap.update_layout(
    title="Participation Density Across Days and Tracks",
    xaxis_title="Track",
    yaxis_title="Day",
    height=400
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Additional statistics
st.subheader("Key Statistics")
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Total Participants", len(df))

with col6:
    avg_rating = df['Rating'].mean()
    st.metric("Average Rating", f"{avg_rating:.1f}")

with col7:
    most_common_track = df['Track'].mode()[0]
    st.metric("Most Popular Track", most_common_track)

with col8:
    most_common_college = df['College'].mode()[0]
    st.metric("Most Represented College", most_common_college)

# Create and display the dashboard
dashboard = Dashboard(df)
dashboard.display() 