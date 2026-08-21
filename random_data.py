import pandas as pd
import numpy as np

np.random.seed(42)

expected_columns = ['VENTA_ZONA_101', 'VENTA_ZONA_102', 'VENTA_ZONA_103', 'VENTA_ZONA_104', 'VENTA_ZONA_107', 'VENTA_ZONA_109', 'VENTA_ZONA_110', 'VENTA_ZONA_111', 'VENTA_ZONA_112', 'VENTA_ZONA_115', 'VENTA_ZONA_116', 'VENTA_ZONA_119', 'PRECIO_NAC', 'N° ASESORAS_ZONA_101', 'N° ASESORAS_ZONA_102', 'N° ASESORAS_ZONA_103', 'N° ASESORAS_ZONA_104', 'N° ASESORAS_ZONA_107', 'N° ASESORAS_ZONA_109', 'N° ASESORAS_ZONA_110', 'N° ASESORAS_ZONA_111', 'N° ASESORAS_ZONA_112', 'N° ASESORAS_ZONA_115', 'N° ASESORAS_ZONA_116', 'N° ASESORAS_ZONA_119', 'N° ASESORAS', 'TALLA_T-12', 'TALLA_T-14', 'TALLA_T-16', 'TALLA_T-6', 'TALLA_T-8', 'TALLA_T-L', 'TALLA_T-M', 'TALLA_T-S', 'TALLA_T-UNI', 'TALLA_T-XL', 'TALLA_T-XS', 'TALLA_T-XXL', 'NOMB_SUBGRUPO_201 RE-BLUSAS FEM', 'NOMB_SUBGRUPO_202 RE-BODYS FEM', 'NOMB_SUBGRUPO_204 RE-BUZOS FEM', 'NOMB_SUBGRUPO_206 RE-CAMISAS FEM', 'NOMB_SUBGRUPO_207 RE-CAMISETAS FEM', 'NOMB_SUBGRUPO_209 RE-CAPRIS FEM', 'NOMB_SUBGRUPO_210 RE-CHALECOS FEM', 'NOMB_SUBGRUPO_211 RE-CHAQUETAS FEM', 'NOMB_SUBGRUPO_213 RE-CONJUNTOS FEM', 'NOMB_SUBGRUPO_214 RE-ENTERIZOS FEM', 'NOMB_SUBGRUPO_215 RE-FALDAS FEM', 'NOMB_SUBGRUPO_216 RE-JEANS FEM', 'NOMB_SUBGRUPO_217 RE-JOGGERS FEM', 'NOMB_SUBGRUPO_218 RE-LEGGINS FEM', 'NOMB_SUBGRUPO_219 RE-OVEROLES FEM', 'NOMB_SUBGRUPO_220 RE-PANTALONES FEM', 'NOMB_SUBGRUPO_221 RE-PESCADORES FEM', 'NOMB_SUBGRUPO_223 RE-SHORTS FEM', 'NOMB_SUBGRUPO_224 RE-SOBRETODOS FEM', 'NOMB_SUBGRUPO_225 RE-VESTIDOS FEM', 'CAMPANA_201902', 'CAMPANA_201903', 'CAMPANA_201904', 'CAMPANA_201905', 'CAMPANA_201906', 'CAMPANA_201907', 'CAMPANA_201908', 'CAMPANA_201909', 'CAMPANA_201910', 'CAMPANA_201911', 'CAMPANA_201912']

# --- Agrupar columnas por tipo ---
zonas = ['101', '102', '103', '104', '107', '109', '110', '111', '112', '115', '116', '119']
venta_cols = [f'VENTA_ZONA_{z}' for z in zonas]
asesoras_zona_cols = [f'N° ASESORAS_ZONA_{z}' for z in zonas]
talla_cols = [c for c in expected_columns if c.startswith('TALLA_')]
subgrupo_cols = [c for c in expected_columns if c.startswith('NOMB_SUBGRUPO_')]
campana_cols = [c for c in expected_columns if c.startswith('CAMPANA_')]

def generar_fila():
    fila = {}

    # Ventas por zona (montos continuos, algunas zonas con más volumen que otras)
    for z in zonas:
        base = np.random.uniform(5000, 60000)
        fila[f'VENTA_ZONA_{z}'] = round(base, 2)

    # Precio nacional promedio de la campaña
    fila['PRECIO_NAC'] = round(np.random.uniform(35, 120), 2)

    # N° asesoras por zona (enteros)
    for z in zonas:
        fila[f'N° ASESORAS_ZONA_{z}'] = int(np.random.randint(10, 200))

    # N° asesoras total = suma de las zonas
    fila['N° ASESORAS'] = sum(fila[f'N° ASESORAS_ZONA_{z}'] for z in zonas)

    # TALLA: one-hot, se elige una talla al azar
    for c in talla_cols:
        fila[c] = 0
    fila[np.random.choice(talla_cols)] = 1

    # NOMB_SUBGRUPO: one-hot, se elige un subgrupo al azar
    for c in subgrupo_cols:
        fila[c] = 0
    fila[np.random.choice(subgrupo_cols)] = 1

    # CAMPANA: one-hot, se elige una campaña al azar
    for c in campana_cols:
        fila[c] = 0
    fila[np.random.choice(campana_cols)] = 1

    return fila

# --- Generar N filas de prueba ---
N = 10
datos = pd.DataFrame([generar_fila() for _ in range(N)])

# Reordenar columnas exactamente como se espera
datos = datos[expected_columns]

print(datos.head())
print("\nForma del dataset:", datos.shape)

# Guardar a CSV
datos.to_csv('run_deployment.py.csv', index=False)
print("\nArchivo guardado en /mnt/user-data/outputs/datos_prediccion_simulados.csv")