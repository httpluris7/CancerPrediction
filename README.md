<!-- FONDO Y TÍTULO -->
<div align="center">
  <h1>🧬 Predicción de Supervivencia en Pacientes con Cáncer</h1>
  <p>
    <strong>Modelo de Machine Learning para estimar el estado vital de pacientes basado en variables clínicas y genéticas.</strong>
  </p>

  <!-- BADGES -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&style=for-the-badge" />
    <img src="https://img.shields.io/badge/Scikit--Learn-green?logo=scikit-learn&style=for-the-badge" />
    <img src="https://img.shields.io/badge/Pandas-yellow?logo=pandas&style=for-the-badge" />
    <img src="https://img.shields.io/badge/Streamlit-ff4b4b?logo=streamlit&style=for-the-badge" />
    <img src="https://img.shields.io/badge/XGBoost-orange?logo=xgboost&style=for-the-badge" />
  </p>
</div>

<br />

<!-- AUTOR -->
### 👤 Autor  
- [Luis Manuel](https://github.com/httpluris7)

---

## 📋 Resumen del Proyecto

Este proyecto se enfoca en desarrollar un modelo de Machine Learning capaz de predecir si un paciente con cáncer está vivo o fallecido, basándose en un conjunto de variables clínicas, demográficas y genéticas sintéticas provenientes de una base de datos simulada.

El enfoque busca apoyar decisiones médicas, priorizando la detección temprana de pacientes en situación de riesgo alto mediante algoritmos de clasificación binaria optimizados.

---

## 🔧 Herramientas Utilizadas

| Herramienta       | Propósito                                      |
|-------------------|-----------------------------------------------|
| **Python**        | Desarrollo de lógica y manipulación de datos  |
| **Pandas**        | Limpieza y transformación de datos            |
| **Scikit-learn**  | Modelado y evaluación de algoritmos ML        |
| **XGBoost**       | Modelo de boosting de alto rendimiento        |
| **Streamlit**     | Visualización interactiva y predicción web    |

---

## 🧠 Flujo de Trabajo

1. **Carga y Preprocesamiento**
   - Limpieza de valores nulos.
   - Codificación de variables categóricas.
   - Escalado y balanceo con SMOTE.

2. **Modelado y Comparación**
   - Entrenamiento de modelos: KNN, Random Forest, Gradient Boosting, AdaBoost y XGBoost.
   - Evaluación mediante métricas: Accuracy, Precision, Recall y F1-Score.
   - Comparación con y sin SMOTE.

3. **Optimización**
   - Uso de GridSearchCV para ajustar hiperparámetros de XGBoost.
   - Selección del mejor modelo según sensibilidad clínica.

4. **Aplicación web**
   - Desarrollo de una app en Streamlit con formulario interactivo de predicción.

---

## 🩺 Resultados Destacados

📌 El modelo **AdaBoost con SMOTE** fue seleccionado como el más adecuado, logrando:
- **Recall del 85.7%** para la clase 'Deceased'.
- **F1-Score del 61.9%**, el más alto entre todos los modelos.
- **Interpretabilidad** y eficiencia computacional aceptable.

El modelo fue implementado en una app visual para facilitar su uso en contextos clínicos o educativos.

---

## 💡 Recomendaciones Finales

✔️ Utilizar AdaBoost con SMOTE para identificar pacientes en riesgo.  
✔️ Enfocar futuras mejoras en agregar más datos reales o variables de seguimiento.  
✔️ Extender la validación con datasets reales o cruzados con historial clínico.  
✔️ Desplegar el modelo como API médica con controles éticos y de seguridad.

---

<!-- FOOTER -->
<div align="center">
  <p>Made with ❤️ by Luis Manuel</p>
</div>
