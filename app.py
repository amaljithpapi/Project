import streamlit as st
import pandas as pd
import pickle
from gtts import gTTS

# Load the trained model
@st.cache_resource
def load_model():
    with open('lr_trained_model (2).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to collect user input
def user_input_features():
    st.sidebar.header("Enter Customer Details")
    
    AGE = st.sidebar.slider("Age", 1, 110, 30)
    
    SEX = st.sidebar.radio("Sex", ["Female", "Male"])
    SEX = 0 if SEX == "Female" else 1
    
    CATEGORY_NAME = st.sidebar.selectbox("Surgery Category", list(range(12)))  # Assuming categories are 0-11
    
    PREAUTH_AMT = st.sidebar.number_input("Pre-Authorization Amount", min_value=0.0, max_value=1000000.0, value=10000.0)

    HOSP_TYPE = st.sidebar.selectbox("Hospital Type", ["Government", "Private"])
    HOSP_TYPE = 0 if HOSP_TYPE == "Government" else 1
    
    
    Mortality = st.sidebar.radio("Mortality", ["No", "Yes"])
    Mortality = 0 if Mortality == "No" else 1
    
    DAYS_STAYED = st.sidebar.number_input("Days Stayed in Hospital", min_value=0, max_value=365, value=3)
    
    # Creating a DataFrame for user input
    data = {
        'AGE': AGE,
        'SEX': SEX,
        'CATEGORY_NAME': CATEGORY_NAME,
        'PREAUTH_AMT': PREAUTH_AMT,
        'HOSP_TYPE': HOSP_TYPE,
        'Mortality': Mortality,
        'DAYS_STAYED': DAYS_STAYED
    }
    
    features = pd.DataFrame(data, index=[0])
    
    # Ensure the features are in the correct order that the model was trained on
    correct_feature_order = ['AGE', 'SEX', 'CATEGORY_NAME','PREAUTH_AMT', 'HOSP_TYPE', 'Mortality', 'DAYS_STAYED']
    ordered_features = features[correct_feature_order]
    
    return ordered_features

# Main function to display the app
def main():
    st.title("💼 Insurance Claim Amount Prediction")
    st.markdown("""
    ### Predict the insurance claim amount based on customer details.
    Please provide the information in the sidebar, and click the button below to see the prediction.
    """)
    
    # Load model
    model = load_model()
    
    # Get user input
    inputs = user_input_features()
    
    # Display input data
    st.subheader("User Input Data")
    st.write(inputs)
    
    # Initialize prediction variable
    prediction = None
    
    # Make prediction when the button is clicked
    if st.button("💡 Predict Claim Amount"):
        try:
            prediction = model.predict(inputs)
            st.success(f"💵 Predicted Claim Amount: ₹{prediction[0]:,.2f}")
            
            # Generate audio only after prediction
            result_text = f'The predicted insurance claim amount is ₹{prediction[0]:,.2f}.'
            tts = gTTS(text=result_text, lang='en')
            tts.save('result_audio.mp3')
            
            # Play the generated audio
            st.write("Audio generated successfully! Click play to listen:")
            audio_file = open('result_audio.mp3', 'rb')
            st.audio(audio_file.read(), format='audio/mp3')
            
        except ValueError as e:
            st.error(f"Prediction Error: {e}")

    # Footer
    st.markdown("""
        ---
        **Created by Amaljith**  
        Developed to predict insurance claim amounts.
    """)

# Run the app
if __name__ == "__main__":
    main()
