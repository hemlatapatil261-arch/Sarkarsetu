import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="SarkarSetu", page_icon="🌉", layout="wide")

@st.cache_resource
def load():
    return joblib.load("models/Random_Forest (1) .pkl"), joblib.load("models/columns.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/schemes.csv"), pd.read_csv("data/users_labeled.csv")

model, cols = load()
sch, df = load_data()
names = sch["name"].tolist()

st.title("🌉 SarkarSetu")
st.caption("Aapka haq, aapke paas | Independent project, official government app nahi")

tab1, tab2, tab3 = st.tabs(["🔍 Find Schemes", "📊 Analytics", "ℹ️ About"])

with tab1:
    st.subheader("Apni jaankari bharo")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 0, 100, 25)
    income = c2.number_input("Saalana income (₹)", 0, 5000000, 150000, step=10000)
    family = c3.number_input("Family size", 1, 15, 4)
    c4, c5, c6, c7 = st.columns(4)
    category = c4.selectbox("Category", ["General", "OBC", "SC", "ST"])
    gender = c5.selectbox("Gender", ["Male", "Female"])
    state = c6.selectbox("State", sorted(df["state"].unique()))
    occ = c7.selectbox("Occupation", sorted(df["occupation"].unique()))

    if st.button("Schemes dhundo", type="primary"):
        row = pd.DataFrame(0, index=[0], columns=cols)
        row["age"], row["income"], row["family_size"] = age, income, family
        for col, val in [("category", category), ("gender", gender),
                         ("state", state), ("occupation", occ)]:
            key = f"{col}_{val}"
            if key in row.columns:
                row[key] = 1

        probs = []
        for est in model.estimators_:
            p = est.predict_proba(row)[0]
            probs.append(p[1] if len(p) > 1 else 0.0)

        res = pd.DataFrame({"Scheme": names, "Match": probs})
        res = res.merge(sch[["name", "documents"]], left_on="Scheme", right_on="name")
        res = res[res["Match"] >= 0.5].sort_values("Match", ascending=False)

        if res.empty:
            st.warning("Is profile ke liye abhi koi scheme match nahi hui.")
        else:
            st.success(f"{len(res)} scheme(s) mil gayi!")
            for _, r in res.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {r['Scheme']}")
                    st.progress(float(r["Match"]), text=f"Match: {r['Match']*100:.0f}%")
                    st.write("📄 **Documents:**", r["documents"])
            st.link_button("Official portal (myScheme)", "https://www.myscheme.gov.in")

with tab2:
    df["num_eligible"] = df[names].sum(axis=1)
    k1, k2, k3 = st.columns(3)
    k1.metric("Total users", len(df))
    k2.metric("Avg schemes/user", f"{df['num_eligible'].mean():.2f}")
    k3.metric("Koi scheme nahi", f"{(df['num_eligible']==0).mean()*100:.0f}%")

    reach = df[names].sum().sort_values().reset_index()
    reach.columns = ["Scheme", "Eligible Users"]
    st.plotly_chart(px.bar(reach, x="Eligible Users", y="Scheme", orientation="h",
                           title="Scheme-wise reach"), use_container_width=True)
    a, b = st.columns(2)
    st_avg = df.groupby("state")["num_eligible"].mean().reset_index()
    a.plotly_chart(px.bar(st_avg, x="state", y="num_eligible",
                          title="State-wise avg schemes"), use_container_width=True)
    cat_avg = df.groupby("category")["num_eligible"].mean().reset_index()
    b.plotly_chart(px.pie(cat_avg, names="category", values="num_eligible",
                          title="Category-wise share"), use_container_width=True)
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv("models/results.csv"), use_container_width=True)

with tab3:
    st.markdown("""
**SarkarSetu** ek ML-based scheme finder hai.

- **Data:** 5000 synthetic users, 15 schemes (illustrative rules)
- **Model:** Random Forest vs Logistic Regression (multi-label)
- **Future scope:** Voice input, OCR, RAG chatbot, real scheme data, regional languages

⚠️ Ye independent academic project hai. Final eligibility ke liye official portal dekho.
""")