import gradio as gr
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "models", "classifier.pkl"), "rb") as f:
    classifier, le = pickle.load(f)
    
with open(os.path.join(BASE_DIR, "models", "regressor.pkl"), "rb") as f:
    regressor = pickle.load(f)

def predict(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_encoded = classifier.predict(features)[0]
    crop_name = le.inverse_transform([crop_encoded])[0]
    yield_pred = regressor.predict(features)[0]
    return crop_name.upper(), f"{round(float(yield_pred), 2)} units"

css = """
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Lato:wght@400;600;700&display=swap');

body, .gradio-container {
    background-color: #ffffff !important;
    font-family: 'Lato', sans-serif !important;
}

.gradio-container {
    max-width: 920px !important;
    margin: auto !important;
}

.block, .form, .wrap, .gap, div[class*="block"], .panel {
    background: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}

/* Fix invisible labels and text */
label,
.label-wrap span,
span.text-gray-500,
span.text-sm,
[data-testid="block-label"],
[data-testid="block-label"] *,
.gradio-slider label,
.gradio-textbox label,
.gradio-number label,
.input-wrap span,
input[type=number] {
    color: #2d2d2d !important;
    opacity: 1 !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 600 !important;
}

/* Slider labels */
[data-testid="block-label"] {
    color: #2d2d2d !important;
    font-size: 0.9em !important;
    font-weight: 600 !important;
}

/* Slider min/max values */
input + div,
span {
    color: #666666;
}

/* Hide only reset buttons */
button.reset-button,
button[aria-label="Reset"],
button svg[data-icon="rotate-left"] {
    display: none !important;
}

/* Numeric inputs */
input[type=number] {
    background: #ffffff !important;
    border: 1px solid #ddd !important;
    color: #2d2d2d !important;

    appearance: auto !important;
    -webkit-appearance: auto !important;
    -moz-appearance: auto !important;
}

/* Keep spinner arrows visible */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    opacity: 1 !important;
    display: block !important;
}

#header {
    background: linear-gradient(135deg, #2d5a1b, #5a3a1a);
    border-radius: 12px;
    padding: 36px 40px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 4px 16px rgba(45,90,27,0.15);
}

#header h1 {
    color: #ffffff !important;
    font-family: 'Merriweather', serif !important;
    font-size: 2.1em !important;
    font-weight: 700 !important;
    margin: 0 0 10px 0 !important;
}

#header p {
    color: #c8ddb8 !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.95em !important;
    margin: 0 !important;
}

.section-title {
    color: #5a3a1a;
    font-family: 'Lato', sans-serif;
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e8ddd0;
}

label {
    color: #2d2d2d !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
}

#predict-btn {
    background: linear-gradient(135deg, #2d5a1b, #3d7a25) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.95em !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 13px !important;
    margin-top: 16px !important;
    width: 100% !important;
    cursor: pointer !important;
}

#predict-btn:hover {
    opacity: 0.88 !important;
}

/* RESULT BOXES */
#result-crop textarea,
#result-yield textarea,
#result-crop input,
#result-yield input {
    background: #f2f7ee !important;
    border: 2px solid #2d5a1b !important;
    border-radius: 8px !important;

    font-family: 'Merriweather', serif !important;
    font-weight: 700 !important;
    color: #2d5a1b !important;

    text-align: center !important;
    padding: 0 !important;
    resize: none !important;
    overflow: hidden !important;
}

/* Crop result */
#result-crop textarea,
#result-crop input {
    font-size: 1.5em !important;
    height: 64px !important;
    min-height: 64px !important;
    max-height: 64px !important;
    line-height: 64px !important;
}

/* Yield result */
#result-yield textarea,
#result-yield input {
    font-size: 1.15em !important;
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    line-height: 52px !important;
}

.info-box {
    margin-top: 20px;
    padding: 16px 18px;
    background: #faf6f0;
    border-radius: 8px;
    border-left: 4px solid #5a3a1a;
}

.info-box p {
    margin: 0;
    color: #555;
    font-family: 'Lato', sans-serif;
    font-size: 0.84em;
    line-height: 1.7;
}

.info-box strong {
    color: #5a3a1a;
}
"""

with gr.Blocks(theme=gr.themes.Default(), css=css, title="Crop Prediction System") as app:

    gr.HTML("""
        <div id="header">
            <h1>Crop Prediction System</h1>
            <p>Enter soil and climate conditions to get a crop recommendation and yield estimate</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML('<div class="section-title">Soil Nutrients</div>')
            N = gr.Slider(0, 140, value=50, label="Nitrogen (N)")
            P = gr.Slider(5, 145, value=50, label="Phosphorus (P)")
            K = gr.Slider(5, 205, value=50, label="Potassium (K)")
            ph = gr.Slider(3.5, 10.0, value=6.5, step=0.1, label="Soil pH")

            gr.HTML('<div class="section-title" style="margin-top:24px">Climate Conditions</div>')
            temperature = gr.Slider(8.0, 44.0, value=25.0, step=0.1, label="Temperature (°C)")
            humidity = gr.Slider(14.0, 100.0, value=70.0, step=0.1, label="Humidity (%)")
            rainfall = gr.Slider(20.0, 300.0, value=100.0, step=0.1, label="Rainfall (mm)")

            btn = gr.Button("🔍  Predict", elem_id="predict-btn")

        with gr.Column(scale=2):
            gr.HTML('<div class="section-title">Results</div>')
            crop_output = gr.Textbox(label="Recommended Crop", elem_id="result-crop", interactive=False)
            yield_output = gr.Textbox(label="Estimated Yield", elem_id="result-yield", interactive=False)
            gr.HTML("""
                <div class="info-box">
                    <p>
                        <strong>How it works</strong><br>
                        The model analyzes soil nutrients (N, P, K), pH,
                        temperature, humidity and rainfall to recommend the best
                        crop and estimate productivity using XGBoost trained on
                        2,200 agricultural samples.
                    </p>
                </div>
            """)

    btn.click(fn=predict, inputs=[N, P, K, temperature, humidity, ph, rainfall],
              outputs=[crop_output, yield_output])

if __name__ == "__main__":
    app.launch()