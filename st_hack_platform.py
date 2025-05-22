import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from datetime import datetime,timedelta
# from langchain.llms import OpenAI
#from langchain_community.llms import OpenAI
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Firebase configuration from Arduino code




# Firebase Realtime Database REST API URL
FIREBASE_REST_URL = f"{DATABASE_URL}{DATA_PATH}.json?auth={API_KEY}"

# Streamlit configuration
st.set_page_config(
    page_title="Smart Rainfall Monitor Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fetch data from Firebase using REST API
@st.cache_data(ttl=60)
def fetch_data():
    try:
        response = requests.get(FIREBASE_REST_URL)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Convert the data to a DataFrame
        if data:
            df = pd.DataFrame.from_dict(data, orient='index')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df = pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Generate dummy data about Tangier if insufficient data
def generate_dummy_data():
    # Create dummy data for Tangier
    dummy_data = {
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
        'waterLevel1': [30 + i * 2 for i in range(10)],
        'waterLevel2': [40 - i * 1 for i in range(10)],
        'RELAY_PIN': [1 if i % 2 == 0 else 0 for i in range(10)]
    }
    return pd.DataFrame(dummy_data)

# Main dashboard layout
#col1, col2 = st.columns([3, 1])

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    time_range = st.selectbox("Time Range", ["Last 24h", "Last 7 Days", "Last 30 Days", "All Data"])
    refresh_rate = st.slider("Auto-refresh (seconds)", 0, 300, 15)
    if st.button("Refresh Data"):
        st.cache_data.clear()

# Initialize data
df = fetch_data()

# Check if data is insufficient and use dummy data
if len(df) < 2:
    st.warning("Insufficient data..") #Using sample data about Tangier
    df = generate_dummy_data()

# Check if data is empty
if df.empty:
    st.warning("No data available. Please check your Firebase configuration.")
else:
    # Dashboard components
    #with col1:
    st.title("🌿 Smart Irrigation System Dashboard")
    
    # Time series visualization
    st.subheader("Underground Tanks Filling Level Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['waterLevel1'], name='Sensor 1'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['waterLevel2'], name='Sensor 2'))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Tank Filling Level (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
        
    # Relay status distribution
    #st.subheader("Relay Activation Patterns")
    #relay_status = df['RELAY_PIN'].apply(lambda x: 'On' if x else 'Off')
    #fig_pie = px.pie(names=relay_status.value_counts().index, values=relay_status.value_counts(), title='Relay Activation Distribution')
    #st.plotly_chart(fig_pie, use_container_width=True)
    
    
    #with col2:
    st.title("📊 Data Analysis")
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    # Data summary for LLM context
    data_summary = f"""
    Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}
    Average Tank 1 fillness level: {df['waterLevel1'].mean():.2f}%
    Average Tank 2 fillness Level: {df['waterLevel2'].mean():.2f}%
    
    """
    #Relay Activation Rate: {relay_status.value_counts().get('On', 0)/len(relay_status):.2%}
    # Analysis chat interface
    analysis_question = st.text_input("Ask about your data:", "What could you deduce from all the data based on your knowledge on the field? Please enlighten us")
    user_question = "What could you deduce from all the data based on your knowledge on the field? Please enlighten us"
    if st.button("Analyze") or analysis_question:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Create prompt as a simple Python string
        system_prompt = """You are an expert agricultural analyst specializing in irrigation systems and drought crisis. We are analyzing rainfall data in Tangier, Morocco within the following context:

        Morocco is experiencing severe water scarcity, with per-capita water availability plummeting from 2,560 m³/year in 1960 to under 620 m³ in 2020, and projected to fall below 500 m³ by 2030 (World Bank). Meanwhile, over 80 % of rainwater in urban areas is wasted—flowing into sewers or the sea—instead of being captured for reuse. This mismanagement causes urban flooding, accelerates groundwater depletion (up to 3 m/year in some regions), and increases reliance on costly water transport. There is an urgent need to capture and repurpose rainwater to supply regions with low annual rainfall.
        
        Provide scientifically accurate analysis focused on:
        - **Key observations** about Tangier’s rainfall patterns and their implications
        - **Pattern recognition** in seasonal and long-term precipitation trends
        - **Actionable recommendations** for urban and peri-urban rainwater harvesting, storage, and distribution
        - **Relevant agricultural science references** to support your conclusions
        """
        
        user_prompt = f"""Data Context:
        {data_summary}
        
        User Question: {analysis_question}"""
        
        payload = {
            "model": "tngtech/deepseek-r1t-chimera:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "stream": False
        }
        full_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        #analysis_chain = LLMChain(llm=llm, prompt=prompt)
        
        with st.spinner("Analyzing data..."):
            try:
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                response = client.chat.completions.create(
                    model="tngtech/deepseek-r1t-chimera:free",
                    messages=full_messages,
                   # max_tokens=1000000,  # Adjust for desired length
                    stream=False
                )
                #response.raise_for_status()
                #result = response.json()
                # answer = result['choices'][0]['message']['content']
                answer = response.choices[0].message.content
                
                st.session_state.analysis_history.append({
                    'question': analysis_question,
                    'answer': answer,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Display analysis history
    for entry in reversed(st.session_state.analysis_history):
        with st.expander(f"Q: {entry['question']} - {entry['timestamp'].strftime('%H:%M')}", expanded=False):
            st.markdown(f"**A:** {entry['answer']}")

# Auto-refresh configuration - Uncomment to enable auto-refresh
# if refresh_rate > 0:
#     time.sleep(refresh_rate)
#     st.experimental_rerun()
