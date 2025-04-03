import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Constants
NUM_PARTICIPANTS = 400
NUM_DAYS = 4
TRACKS = ['Technical', 'Research', 'Innovation', 'Design']
TIME_SLOTS = ['9:00 AM - 10:30 AM', '11:00 AM - 12:30 PM', 
              '2:00 PM - 3:30 PM', '4:00 PM - 5:30 PM']

# Sample data for more realistic generation
COLLEGES = [
    'MIT', 'Stanford University', 'Harvard University', 'Caltech',
    'University of California Berkeley', 'Carnegie Mellon University',
    'Georgia Institute of Technology', 'University of Michigan',
    'University of Illinois Urbana-Champaign', 'Purdue University',
    'University of Texas at Austin', 'University of Wisconsin-Madison',
    'University of Washington', 'University of California Los Angeles',
    'University of Pennsylvania'
]

STATES = [
    'California', 'Massachusetts', 'New York', 'Texas', 'Illinois',
    'Michigan', 'Georgia', 'Pennsylvania', 'Washington', 'Wisconsin',
    'Indiana', 'New Jersey', 'Virginia', 'North Carolina', 'Ohio'
]

# Generate feedback templates
FEEDBACK_TEMPLATES = [
    "Excellent presentation with clear methodology and results.",
    "Good research work, but could improve on visual presentation.",
    "Innovative approach to the problem, well-executed study.",
    "Strong technical foundation, but could benefit from more real-world applications.",
    "Impressive research design and methodology.",
    "Well-structured presentation with good use of visuals.",
    "Interesting topic with practical implications.",
    "Good technical depth, but could improve on communication.",
    "Novel approach to solving the problem.",
    "Comprehensive study with clear findings."
]

def generate_poster_title(track):
    """Generate a realistic poster title based on the track."""
    if track == 'Technical':
        prefixes = ['Development of', 'Implementation of', 'Analysis of']
        topics = ['Machine Learning Algorithms', 'Cloud Computing Systems', 
                 'Blockchain Technology', 'IoT Solutions', 'Cybersecurity Framework']
    elif track == 'Research':
        prefixes = ['Study on', 'Investigation of', 'Analysis of']
        topics = ['Climate Change Impact', 'Renewable Energy Solutions', 
                 'Healthcare Systems', 'Educational Technology', 'Urban Development']
    elif track == 'Innovation':
        prefixes = ['Novel Approach to', 'Innovative Solution for', 'Breakthrough in']
        topics = ['Sustainable Energy', 'Smart Cities', 'Digital Healthcare', 
                 'Agricultural Technology', 'Environmental Protection']
    else:  # Design
        prefixes = ['Design of', 'Development of', 'Creation of']
        topics = ['User Interface', 'Sustainable Architecture', 'Product Design', 
                 'Urban Planning', 'Interactive Systems']
    
    return f"{random.choice(prefixes)} {random.choice(topics)}"

def generate_feedback():
    """Generate realistic feedback."""
    base_feedback = random.choice(FEEDBACK_TEMPLATES)
    additional_comments = [
        " The presenter demonstrated strong technical knowledge.",
        " The methodology was well-explained.",
        " The results were presented clearly.",
        " The visual aids were effective.",
        " The Q&A session was handled well.",
        " The research implications were well-discussed.",
        " The practical applications were well-explained.",
        " The future work direction was clear.",
        " The presentation style was engaging.",
        " The technical depth was appropriate."
    ]
    return base_feedback + random.choice(additional_comments)

# Generate data
data = {
    'Participant_ID': [f'P{i:03d}' for i in range(1, NUM_PARTICIPANTS + 1)],
    'Name': [fake.name() for _ in range(NUM_PARTICIPANTS)],
    'College': [random.choice(COLLEGES) for _ in range(NUM_PARTICIPANTS)],
    'State': [random.choice(STATES) for _ in range(NUM_PARTICIPANTS)],
    'Track': [random.choice(TRACKS) for _ in range(NUM_PARTICIPANTS)],
    'Day': [random.randint(1, NUM_DAYS) for _ in range(NUM_PARTICIPANTS)],
    'Time_Slot': [random.choice(TIME_SLOTS) for _ in range(NUM_PARTICIPANTS)],
    'Poster_Title': [generate_poster_title(random.choice(TRACKS)) for _ in range(NUM_PARTICIPANTS)],
    'Feedback': [generate_feedback() for _ in range(NUM_PARTICIPANTS)],
    'Rating': [round(random.uniform(3.0, 5.0), 1) for _ in range(NUM_PARTICIPANTS)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Day and Time Slot for better organization
df = df.sort_values(['Day', 'Time_Slot'])

# Save to CSV
df.to_csv('poster_presentation_data.csv', index=False)
print("Dataset has been generated and saved as 'poster_presentation_data.csv'") 