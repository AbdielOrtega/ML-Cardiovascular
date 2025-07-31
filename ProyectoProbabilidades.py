import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Cargar la base de datos
df = pd.read_excel("Posiblespropuestas.xlsx")

# Revisar columnas


# Convertir variable categórica 'famhist' a numérica
df['famhist'] = df['famhist'].map({'Present': 1, 'Absent': 0})

# ====================
# GRÁFICAS EXPLORATORIAS
# ====================
# Distribución de edades
plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, color='teal')
plt.title("Distribución de Edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# Frecuencia de enfermedad CHD
plt.figure(figsize=(4,4))
sns.countplot(x='chd', data=df, palette='Set2')
plt.title("Frecuencia de Enfermedad Coronaria (CHD)")
plt.xlabel("Tiene CHD (0 = No, 1 = Sí)")
plt.ylabel("Número de Pacientes")
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparación LDL según CHD
plt.figure(figsize=(6,4))
sns.boxplot(x='chd', y='ldl', data=df, palette='Set3')
plt.title("Niveles de LDL según CHD")
plt.xlabel("CHD")
plt.ylabel("LDL")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================
# MODELO DE REGRESIÓN LOGÍSTICA
# ====================

# Seleccionar todas las variables de entrada menos 'chd' como features
features = [col for col in df.columns if col != 'chd']
X = df[features]
y = df['chd']

# Escalado de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)


# Entrenar el modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


import numpy as np

# Imprimir resultados
print(f"\nPrecisión del modelo: {acc:.4f}")
print("\nReporte de clasificación:")
print(report)



# Permitir al usuario ingresar datos personalizados de forma dinámica y llamativa
import time
print("\n" + "="*60)
print("¡Bienvenido al predictor de riesgo de Enfermedad Coronaria (CHD)!\n")
print("Variables de entrada que debes ingresar:")
print("-"*60)
for i, col in enumerate(features, 1):
    if col == 'famhist':
        print(f"{i}. {col} (Present/Absent)")
    else:
        print(f"{i}. {col}")
print("-"*60)
input("Presiona ENTER para comenzar a ingresar los datos...\n")

try:
    entrada = []
    for col in features:
        mensaje = f"\nIngresa el valor para '{col}': "
        if col == 'famhist':
            mensaje = f"\n¿Antecedentes familiares de enfermedad coronaria? (Present/Absent): "
            val = input(mensaje).strip()
            val = 1 if val.lower().startswith('p') else 0
        else:
            val = input(mensaje)
            val = float(val)
        entrada.append(val)
        print(f"✔️  {col} registrado!")
        time.sleep(0.3)
    print("\nProcesando predicción... 🚀\n")
    time.sleep(1)
    paciente_usuario = np.array([entrada])
    paciente_usuario_scaled = scaler.transform(paciente_usuario)
    prob_usuario = model.predict_proba(paciente_usuario_scaled)[0][1] * 100
    print("="*60)
    if prob_usuario >= 50:
        print(f"🔴 Riesgo ALTO de CHD: {prob_usuario:.2f}%")
    else:
        print(f"🟢 Riesgo BAJO de CHD: {prob_usuario:.2f}%")
    print("="*60)
except Exception as e:
    print(f"Error en la entrada de datos: {e}")