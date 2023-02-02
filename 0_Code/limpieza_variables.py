# -*- coding: utf-8 -*-
"""
Código para la limpieza de las distintas variables de la base de Tropa

Elaborado por: Daniel Cárdenas, Jenny Rivera y Laura Rodriguez
Fecha elaboración: 29/08/2022
"""

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import re


# Función para limpiar los datos de la edad, input:edad, output: edad modificada
def limpieza_edad(edad):
    return edad if edad > 18 and edad < 99 else None

# Función para limpiar los datos de los ingresos, input:ingresos, output: ingresos modificada
def limpieza_ingresos(ingresos):
    if ingresos > 1000000:
        return ingresos
    elif ingresos > 9000000:
        return 9000000
    else:
        None

# Función para limpiar los datos de número de trabajadores, input:número de trabajadores, output: número de trabajadores modificada
def limpieza_num_trabajadores(trabajadores):
    if trabajadores > 0:
        return trabajadores
    elif trabajadores > 10:
        return 10
    else:
        None

def limpieza_num_trabajadores_mayor(trabajadores):
    if trabajadores > 0 and trabajadores < 10:
        return 0
    elif trabajadores >= 10:
        return 1
    else:
        None


def limpieza_antiguedad_mayor(antiguedad):
    return 1 if antiguedad != 'Menos de un año' else 0

def limpieza_antiguedad_mayor_2years(antiguedad):
    return 1 if antiguedad not in ['Menos de un año', 'De 1 a menos de 3 años']  else 0

def corazon_productivo(UPZ):
    return 1 if UPZ in ['Doce_De_Octubre', 'Restrepo', 'Venecia'] else 0

def perfil_crediticio(INTERES_PERFIL):
    return 1 if INTERES_PERFIL in ['Si le interesa', 'Si le interesa el crédito pero'] else 0

def dummy_propietario(propietario):
    return 1 if propietario > 0 else 0


# %% Digitos CIIU
# Función para obtener primeros x digitos del CIIU
def obtener_primeros_digitos(df, col_names,  num):
    return df[col_names].astype(str).str[:num]

# Dummy para grandes secciones de actividades economicas según CIIU
def categorize_ciiu(ciiu):
    if ciiu >= 10 and ciiu <= 33:
         return 'INDUSTRIAS MANUFACTURERAS'
    elif ciiu == 35:
        return 'SUMINISTRO DE ELECTRICIDAD'
    elif ciiu >= 36 and ciiu <= 39:
        return 'DISTRIBUCIÓN DE AGUA Y OTROS'
    elif ciiu >= 41 and ciiu <= 43:
        return 'CONSTRUCCION'
    elif ciiu >= 45 and ciiu <= 47:
        return 'COMERCIO AL POR MAYOR Y AL POR MENOR o REPARACIÓN DE VEHÍCULOS'
    elif ciiu >= 49 and ciiu <= 53:
        return 'TRANSPORTE Y ALMACENAMIENTO'
    elif ciiu >= 55 and ciiu <= 56:
        return 'ALOJAMIENTO Y SERVICIOS DE COMIDA'
    elif ciiu >= 58 and ciiu <= 63:
        return 'INFORMACIÓN Y COMUNICACIONES'
    elif ciiu >= 64 and ciiu <= 66:
        return 'ACTIVIDADES FINANCIERAS Y DE SEGUROS'
    elif ciiu == 68:
        return 'ACTIVIDADES INMOBILIARIAS'
    elif ciiu >= 69 and ciiu <= 75:
        return 'ACTIVIDADES PROFESIONALES CIENTIFICAS Y TECNICAS'
    elif ciiu >= 77 and ciiu <= 82:
        return 'ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO'
    elif ciiu == 84:
        return 'ADMINISTRACIÓN PUBLICA Y DEFENSA'
    elif ciiu == 85:
        return 'EDUCACION'
    elif ciiu >= 86 and ciiu <= 88:
        return 'ACTIVIDADES DE ATENCIÓN DE LA SALUD HUMANA'
    elif ciiu >= 90 and ciiu <= 93:
        return 'ACTIVIDADES ARTISTICAS DE ENTRETENIMIENTO Y RECREACION'
    elif ciiu >= 94 and ciiu <= 96:
        return 'OTRAS ACTIVIDADES DE SERVICIOS'
    else:
            None



# %% Creacion nuevas columnas
# Función para crear nueva columna de respuestas separadas por delim
def crear_nuevas_columnas(df, col_names):
    df_nuevas = df[col_names].str.split(pat = ",", expand=True).apply(lambda x : x.value_counts(), axis = 1).fillna(0).astype(int)
    df_nuevas.columns = [col_names + '_' + str(col)  for col in df_nuevas.columns]
    return pd.concat([df, df_nuevas], axis=1)





