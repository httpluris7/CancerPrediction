
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

modelo_final = joblib.load("modelo_adaboost_smote.pkl")

st.title("🔬 Comparador de Modelos ML y Predicción Individual")

df = pd.read_csv("china_cancer_patients_synthetic.csv")
df['Comorbidities'].fillna('None', inplace=True)
df['GeneticMutation'].fillna('None', inplace=True)
df_ml = df.drop(columns=["PatientID", "DiagnosisDate", "SurgeryDate"])
X = df_ml.drop("SurvivalStatus", axis=1)
y = df_ml["SurvivalStatus"].map({'Alive': 0, 'Deceased': 1})
X_encoded = pd.get_dummies(X, drop_first=True)

glosario = {
    "Age": "Edad del paciente en años.",
    "Gender": "Sexo del paciente (Male o Female).",
    "TumorStage": "Etapa clínica del tumor (Early, Middle, Advanced).",
    "Comorbidities": "Otras enfermedades presentes además del cáncer.",
    "GeneticMutation": "Mutaciones genéticas identificadas (BRCA1, EGFR, etc.).",
    "TreatmentType": "Tipo principal de tratamiento aplicado.",
    "Radiation": "Si recibió o no radiación.",
    "Chemotherapy": "Si recibió o no quimioterapia.",
    "Immunotherapy": "Si recibió o no inmunoterapia.",
    "Hospital": "Hospital donde fue tratado el paciente.",
    "SurvivalStatus": "Estado final del paciente (0=Alive, 1=Deceased)."
}


#Dividimos y escalamos
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Sidebar
st.sidebar.title("⚙️ Opciones de modelo")
apply_smote = st.sidebar.checkbox("Aplicar SMOTE", value=False)
model_choice = st.sidebar.selectbox("Modelo", ["KNN", "Random Forest", "Gradient Boosting", "AdaBoost", "XGBoost"])
k = 13
if model_choice == "KNN":
    k = st.sidebar.slider("Vecinos (KNN)", 1, 30, 13)

#Función para obtener modelo
def get_model(choice):
    if choice == "KNN":
        return KNeighborsClassifier(n_neighbors=k)
    elif choice == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif choice == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif choice == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    elif choice == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


model = get_model(model_choice)

# Aplicar SMOTE si está marcado
X_train_final, y_train_final = X_train_scaled, y_train
if apply_smote:
    smote = SMOTE(random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

# Entrenar modelo seleccionado
model.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_scaled)

# Evaluación
st.subheader("📈 Evaluación del Modelo Seleccionado")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Características importantes
if model_choice in ["Random Forest", "Gradient Boosting"]:
    st.subheader("🔎 Características más importantes")
    importances = model.feature_importances_
    feature_names = X_encoded.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_n = st.slider("¿Cuántas características mostrar?", 5, 30, 10)
    top_features = importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Importance", y="Feature", data=top_features, ax=ax_imp)
    ax_imp.set_title("Top Características Más Importantes")
    st.pyplot(fig_imp)

# Comparación de modelos
st.subheader("📊 Comparar todos los modelos con y sin SMOTE")
def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_te, y_te, usar_smote=False):
    if usar_smote:
        sm = SMOTE(random_state=42)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    modelo.fit(X_tr, y_tr)
    y_pr = modelo.predict(X_te)
    return {
        "Modelo": nombre,
        "SMOTE": usar_smote,
        "Accuracy": accuracy_score(y_te, y_pr),
        "Precision": precision_score(y_te, y_pr, pos_label=1),
        "Recall": recall_score(y_te, y_pr, pos_label=1),
        "F1-Score": f1_score(y_te, y_pr, pos_label=1)

    }

modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=13),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = []
for nombre, modelo in modelos.items():
    resultados.append(evaluar_modelo(nombre, modelo, X_train_scaled, y_train, X_test_scaled, y_test, usar_smote=False))
    resultados.append(evaluar_modelo(nombre, modelo, X_train_scaled, y_train, X_test_scaled, y_test, usar_smote=True))
df_resultados = pd.DataFrame(resultados).sort_values(by="F1-Score", ascending=False)
st.dataframe(df_resultados)

# Sección: Mejor modelo identificado
st.subheader("🏆 Modelo Más Exitoso")

# Extraemos el mejor modelo por F1-Score
mejor_fila = df_resultados.iloc[0]
modelo_top = mejor_fila["Modelo"]
uso_smote = "Sí" if mejor_fila["SMOTE"] else "No"
f1 = mejor_fila["F1-Score"]
recall = mejor_fila["Recall"]
precision = mejor_fila["Precision"]
accuracy = mejor_fila["Accuracy"]

st.markdown(f"""
### 🔹 {modelo_top} (SMOTE: {uso_smote})

**📈 Métricas clave:**
- **F1-Score:** {f1:.3f}
- **Recall (Sensibilidad):** {recall:.3f}
- **Precisión:** {precision:.3f}
- **Accuracy:** {accuracy:.3f}
""")

st.info(f"""
🧠 **¿Por qué es el mejor modelo?**

Este modelo fue seleccionado por obtener el mayor **F1-Score**, lo que indica un excelente equilibrio entre **precisión** y **recall**. 
Además, su alta **sensibilidad (recall)** lo hace ideal para contextos médicos donde es crucial identificar correctamente a los pacientes en riesgo.

En particular, detecta de manera eficaz los casos de fallecimiento (clase 1), minimizando falsos negativos.
""")


# Comparación visual de modelos
st.subheader("📊 Comparación Visual: F1-Score y Recall")

# Gráfico de F1-Score
fig_f1, ax_f1 = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_resultados, x="Modelo", y="F1-Score", hue="SMOTE", ax=ax_f1)
ax_f1.set_title("F1-Score por Modelo (con y sin SMOTE)")
ax_f1.set_ylim(0, 1)
st.pyplot(fig_f1)

# Gráfico de Recall
fig_recall, ax_recall = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_resultados, x="Modelo", y="Recall", hue="SMOTE", ax=ax_recall)
ax_recall.set_title("Recall por Modelo (con y sin SMOTE)")
ax_recall.set_ylim(0, 1)
st.pyplot(fig_recall)



# Predicción individual
st.sidebar.markdown("---")

with st.sidebar.expander("📘 Glosario de Variables"):
    for variable, definicion in glosario.items():
        st.markdown(f"**{variable}**: {definicion}")

st.sidebar.subheader("🔮 Predicción Individual")

input_data = {}
for col in X.columns:
    if df[col].dtype == object:
        input_data[col] = st.sidebar.selectbox(f"{col}", sorted(df[col].dropna().unique()))
    else:
        input_data[col] = st.sidebar.number_input(f"{col}", value=float(df[col].mean()))

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)

if st.sidebar.button("Predecir Resultado"):
    pred = model.predict(input_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]  # probabilidad de clase 1
        resultado = "Deceased" if pred == 1 else "Alive"
        st.sidebar.success(f"✅ {resultado}")
        st.sidebar.info(f"🔎 Prob. fallecimiento: **{prob:.2%}**")
    else:
        st.sidebar.warning("Este modelo no permite estimar probabilidades.")


