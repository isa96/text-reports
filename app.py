import streamlit as st

import text_preprocessing
import model

# Title
st.title("Public Report Classification")

# Load model
with st.spinner("Loading Model...."):
    my_model = model.load_model()

# Get new report
new_report = st.text_area("Ada masalah apa?","")

# Define the CSS style for the cards
card_style = """
    border-style: solid;
    border-width: 2px;
    border-color: #959595;
    border-radius: 10px;
    box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
    padding: 20px;
    margin: 10px;
    height: auto;
    display: flex;
    flex-direction: column;
"""

# threshold accuracy
threshold = 10

if st.button("Prediksi Kategori Laporan", use_container_width=True):
    with st.spinner("Tunggu sebentar, sedang memprediksi kategori laporan..."):
        df, preprocess_text = model.predict(my_model, new_report)
        
        col1, col2 = st.columns(2)
        for index, row in df.iterrows():
            if row['probability'] > threshold:
                if index % 2 == 0:
                    card_col = col1
                else:
                    card_col = col2
                with card_col:
                    st.markdown(
                    f"""
                        <div style="{card_style}">
                            <h3>{row['prediksi_kategori_laporan'].upper()}</h3>
                            <p>Probabilitas: {round(row['probability'], 2)}%</p>
                        </div>
                    """, unsafe_allow_html=True
                    )
        
    st.divider() # Draw a horizontal line
    
    # Show preprocessed text
    with st.expander("Teks hasil preprocessing"):
        st.write(preprocess_text)
        
    # Show table of all categories
    with st.expander("Tabel semua prediksi kategori"):
        st.dataframe(df, use_container_width=True)
