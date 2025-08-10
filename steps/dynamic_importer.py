import pandas as pd
from zenml import step

@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data={ 'VENTA_ZONA_101': 152,
     'VENTA_ZONA_102': 0,
     'VENTA_ZONA_103': 0,
    'VENTA_ZONA_104': 0,
    'VENTA_ZONA_107': 20,
    'VENTA_ZONA_109': 0,
    'VENTA_ZONA_110': 0,
    'VENTA_ZONA_111': 0,
    'VENTA_ZONA_112': 10,
    'VENTA_ZONA_115': 0,
    'VENTA_ZONA_116': 0,
    'VENTA_ZONA_119': 0,
    'PRECIO_NAC': 99900,
    'N° ASESORAS_ZONA_101': 821,
    'N° ASESORAS_ZONA_102': 819,
    'N° ASESORAS_ZONA_103': 662,
    'N° ASESORAS_ZONA_104': 351,
    'N° ASESORAS_ZONA_107': 424,
    'N° ASESORAS_ZONA_109': 467,
    'N° ASESORAS_ZONA_110': 428,
    'N° ASESORAS_ZONA_111': 521,
    'N° ASESORAS_ZONA_112': 417,
    'N° ASESORAS_ZONA_115': 420,
    'N° ASESORAS_ZONA_116': 727,
    'N° ASESORAS_ZONA_119': 419,
    'N° ASESORAS': 25312,
    'TALLA_T-12': 2,
    'TALLA_T-14': 78,
    'TALLA_T-16': 69,
    'TALLA_T-6': 52,
    'TALLA_T-8': 55,
    'TALLA_T-L': 81,
    'TALLA_T-M': 31,
    'TALLA_T-S': 88,
    'TALLA_T-UNI': 4,
    'TALLA_T-XL': 34,
    'TALLA_T-XS': 49,
    'TALLA_T-XXL': 44,
    'NOMB_SUBGRUPO_201 RE-BLUSAS FEM': 32,
    'NOMB_SUBGRUPO_202 RE-BODYS FEM': 59,
    'NOMB_SUBGRUPO_204 RE-BUZOS FEM': 23,
    'NOMB_SUBGRUPO_206 RE-CAMISAS FEM': 15,
    'NOMB_SUBGRUPO_207 RE-CAMISETAS FEM': 43,
    'NOMB_SUBGRUPO_209 RE-CAPRIS FEM': 26,
    'NOMB_SUBGRUPO_210 RE-CHALECOS FEM': 29,
    'NOMB_SUBGRUPO_211 RE-CHAQUETAS FEM': 34,
    'NOMB_SUBGRUPO_213 RE-CONJUNTOS FEM': 58,
    'NOMB_SUBGRUPO_214 RE-ENTERIZOS FEM': 50,
    'NOMB_SUBGRUPO_215 RE-FALDAS FEM': 80,
    'NOMB_SUBGRUPO_216 RE-JEANS FEM': 68,
    'NOMB_SUBGRUPO_217 RE-JOGGERS FEM': 24,
    'NOMB_SUBGRUPO_218 RE-LEGGINS FEM': 99,
    'NOMB_SUBGRUPO_219 RE-OVEROLES FEM': 79,
    'NOMB_SUBGRUPO_220 RE-PANTALONES FEM': 98,
    'NOMB_SUBGRUPO_221 RE-PESCADORES FEM': 98,
    'NOMB_SUBGRUPO_223 RE-SHORTS FEM': 39,
    'NOMB_SUBGRUPO_224 RE-SOBRETODOS FEM': 82,
    'NOMB_SUBGRUPO_225 RE-VESTIDOS FEM': 39,
    'CAMPANA_201902': 75,
    'CAMPANA_201903': 80,
    'CAMPANA_201904': 15,
    'CAMPANA_201905': 50,
    'CAMPANA_201906': 87,
    'CAMPANA_201907': 32,
    'CAMPANA_201908': 76,
    'CAMPANA_201909': 81,
    'CAMPANA_201910': 53,
    'CAMPANA_201911': 91,
    'CAMPANA_201912': 43
    }

    
    df = pd.DataFrame(data, index=[0])
    print(df.shape)  # (1, 69)
    print(df.head())

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data