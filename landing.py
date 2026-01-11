import streamlit as st

# --------- Auth Guard ----------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login to continue")
    st.switch_page("login.py")

import streamlit as st

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="MedCopilot OS — Medical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ================== HEADER ==================
st.markdown("""
# 🧠 MedCopilot OS — Medical Intelligence Platform
### Clinical Care • Medical Research • Trials • Regulatory Intelligence
""")

st.divider()

# ================== PLATFORM INTRO ==================
st.markdown("""
Welcome to **MedCopilot OS**, a unified medical intelligence platform designed for:

- 👨‍⚕️ Doctors & Clinicians  
- 🔬 Medical Researchers  
- 🧪 Clinical Trial Teams  
- 💊 Pharma & Drug Development  
- 🧾 Regulatory & Compliance Teams  

Choose your workspace to continue.
""")

st.divider()

# ================== WORKSPACE SELECTION ==================
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🏥 Hospital Clinical AI")
    st.markdown("""
    Evidence-based clinical decision support system.

    **Features**
    - Hospital protocol intelligence  
    - Diagnosis & treatment guidance  
    - Drug dosage & monitoring  
    - Evidence citations  
    - Hybrid AI reasoning  

    **For**
    - Doctors  
    - Hospitals  
    - Clinical teams  
    """)

    if st.button("🚀 Enter Hospital AI", use_container_width=True):
        st.switch_page("app.py")

with col2:
    st.markdown("## 🔬 Medical Research AI")
    st.markdown("""
    Biomedical research intelligence platform.

    **Features**
    - PubMed literature intelligence  
    - Clinical trials intelligence  
    - Drug discovery research  
    - Treatment comparisons  
    - Research citations  

    **For**
    - Researchers  
    - Pharma companies  
    - Medical colleges  
    """)

    if st.button("🚀 Enter Research AI", use_container_width=True):
        st.switch_page("research_ai/research_dashboard.py")

st.divider()

# ================== FUTURE MODULES ==================
st.markdown("## 🔮 Upcoming Intelligence Modules")

col3, col4, col5 = st.columns(3)

with col3:
    st.info("🧪 Clinical Trials Intelligence\n\nPhase-wise trials, outcomes, enrollment, results")

with col4:
    st.info("💊 FDA Regulatory Intelligence\n\nDrug approvals, safety alerts, indications")

with col5:
    st.info("📊 Analytics & Reports\n\nTreatment trends, drug pipeline, research insights")

# ================== FOOTER ==================
st.divider()
st.caption("MedCopilot OS © Medical Intelligence Platform | Clinical • Research • Regulatory")
