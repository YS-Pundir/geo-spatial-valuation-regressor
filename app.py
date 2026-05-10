import gradio as gr
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained pipeline
# Note: Using relative path 'Export/' to ensure it works on Hugging Face
model_path = "Export/housing_regressor_pipeline.joblib"
pipeline = joblib.load(model_path)

# Feature names from the California Housing Dataset
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

def predict_house_value(med_inc, house_age, rooms, bedrooms, pop, occup, lat, lon):
    # Create a DataFrame for the model
    input_data = pd.DataFrame([[
        med_inc, house_age, rooms, bedrooms, pop, occup, lat, lon
    ]], columns=FEATURES)
    
    # Model returns value in $100,000s
    prediction = pipeline.predict(input_data)[0]
    
    # Convert to standard currency format
    actual_value = prediction * 100000
    
    return f"${actual_value:,.2f}"

# 2. Build the Interface with UE Potsdam & IIT Branding
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Geo-Spatial Housing Valuation Engine")
    gr.Markdown("### **University of Europe for Applied Sciences, Potsdam | IIT Roorkee Capstone**")
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📍 Location & Demographics")
            med_inc = gr.Slider(0, 15, value=3.5, label="Median Income (Units of $10k)")
            house_age = gr.Slider(1, 52, value=28, label="Median House Age")
            pop = gr.Number(value=1400, label="Block Population")
            lat = gr.Slider(32, 42, value=34.0, label="Latitude")
            lon = gr.Slider(-124, -114, value=-118.0, label="Longitude")
            
        with gr.Column():
            gr.Markdown("### 🏗️ Property Characteristics")
            rooms = gr.Slider(1, 10, value=5, label="Avg Rooms per Household")
            bedrooms = gr.Slider(1, 5, value=1, label="Avg Bedrooms per Household")
            occup = gr.Slider(1, 10, value=3, label="Avg Occupants per Household")
            
            gr.Markdown("---")
            result = gr.Textbox(label="Estimated Market Valuation", interactive=False)
            predict_btn = gr.Button("🔍 Run Valuation Analysis", variant="primary")

    gr.Markdown("""
    **Model Insight:** This engine uses a **Tuned Random Forest Regressor** to analyze non-linear 
    relationships between geographic coordinates and economic data.
    """)

    predict_btn.click(
        fn=predict_house_value,
        inputs=[med_inc, house_age, rooms, bedrooms, pop, occup, lat, lon],
        outputs=result
    )

if __name__ == "__main__":
    demo.launch()