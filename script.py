import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('./utils/amostra_isolation_forest.csv', sep=';')
print(df.head())

modelo_cpu_ram = df.copy()
metricas = [
    'CPU_PERCENT',
    'CPU_USER_PERCENT',
    'CPU_SYSTEM_PERCENT',
    'CPU_IDLE_PERCENT','CPU_IOWAIT_PERCENT',
    'CPU_FREQ_ATUAL',
    'CPU_COUNT_LOGICA',
    'LOAD_AVG_1',
    'LOAD_AVG_5',
    'LOAD_AVG_15',
    'RAM_PERCENT',
    'RAM_TOTAL',
    'RAM_AVAILABLE',
    'RAM_USED'
]

x = df[metricas].dropna()

modelo = IsolationForest(
    n_estimators=200, 
    contamination="auto", 
    random_state=42
)

modelo.fit(x)

df.loc[x.index, "anomalia"] = modelo.predict(x)
df.loc[x.index, "score"] = modelo.decision_function(x)

anomalias = df[df["anomalia"] == -1].sort_values("score")
print(anomalias)

df.to_csv("resultado.csv", index=False)

