import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

st.set_page_config(
    page_title="Smart Irrigation AI System",
    layout="wide"
)

@st.cache_data
def load_model():
    """Loads the pre-trained irrigation model."""
    try:
        model = joblib.load('smart_irrigation_model.joblib')
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def get_crop_info():
    """Loads and cleans the crop data to get average moisture levels."""
    try:
        df_core = pd.read_csv('data_core.csv', on_bad_lines='skip')
        df_core.columns = ['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer_Name']
        avg_moisture = df_core.groupby('Crop_Type')['Moisture'].mean().reset_index()
        return avg_moisture
    except FileNotFoundError:
        return None

@st.cache_data
def load_scatter_data():
    """Loads the data for the interactive plot background."""
    try:
        df = pd.read_csv('dataaa.csv')
        return df
    except FileNotFoundError:
        return None

def plot_interactive_scatter(current_moisture, current_temp, df_background):
    """
    Plots the historical data and highlights the current slider position.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=df_background, x='moisture', y='temp', hue='pump',
        palette={0: 'royalblue', 1: 'orangered'},
        ax=ax, style='pump', markers={0: 'o', 1: 's'},
        alpha=0.6, s=80
    )
    ax.scatter(
        x=current_moisture, y=current_temp, color='limegreen',
        marker='*', s=500, edgecolor='black', linewidth=1.5, label='Current Reading'
    )
    ax.set_title('Real-Time Sensor Position vs. Historical Data')
    ax.set_xlabel('Soil Moisture (Raw Sensor Value)')
    ax.set_ylabel('Temperature (°C)')
    handles, labels = ax.get_legend_handles_labels()
    custom_labels = ['Pump Off (Historical)', 'Pump On (Historical)', 'Current Reading']
    ax.legend(handles=handles, labels=custom_labels)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)

model = load_model()
crop_info = get_crop_info()
df_scatter = load_scatter_data()

st.title("Smart Irrigation AI Decision Support System")
st.markdown("This dashboard uses a machine learning model to provide real-time irrigation recommendations.")

if model is None or df_scatter is None:
    st.error("Required file not found. Please ensure 'smart_irrigation_model.joblib' and 'dataaa.csv' are in the correct directory.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Sensor Input Simulation")
        st.markdown("Adjust the sliders to simulate real-time sensor readings from the field.")
        
        temp = st.slider('Temperature (°C)', min_value=0, max_value=50, value=28, step=1)
        moisture = st.slider('Soil Moisture (Raw Sensor Value)', min_value=0, max_value=1100, value=560, step=10)

        new_data = pd.DataFrame([[moisture, temp]], columns=['moisture', 'temp'])
        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)

        st.subheader("AI Recommendation")
        
        cotton_avg_moisture = 50.0 
        if crop_info is not None:
            try:
                cotton_avg_moisture = crop_info[crop_info['Crop_Type'] == 'Cotton']['Moisture'].iloc[0]
            except IndexError:
                pass 
        
        if prediction[0] == 0:
            st.error("ACTION: **TURN PUMP ON**")
            prob_dry = probability[0][0]
            st.write(f"The model is **{prob_dry:.0%}** confident that the soil is dry.")
            if moisture < 300:
                st.warning("Moisture level is critically low. Immediate irrigation is strongly advised.")
        else:
            st.success("ACTION: **KEEP PUMP OFF**")
            prob_wet = probability[0][1]
            st.write(f"The model is **{prob_wet:.0%}** confident that the soil has sufficient moisture.")
        
        if crop_info is not None:

            normalized_moisture = (moisture / 1100) * 100
            st.info(f"Context: The average moisture for Cotton is **~{cotton_avg_moisture:.0f}%**. Your current reading of **{moisture}** is equivalent to **{normalized_moisture:.1f}%**.")

    with col2:
        st.header("Interactive Decision Boundary")
        st.markdown("See where your simulated sensor reading (⭐) falls in relation to the historical data.")
        plot_interactive_scatter(moisture, temp, df_scatter)