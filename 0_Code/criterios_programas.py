criterios_dict = {'Agencia Pública de Empleo':{'METAS_Capacitar_a_su_personal': 1,
                                               'METAS_Incrementar_su_planta_de_person': 1},
                  'Pago por resultados o Empleo Joven':{'METAS_Incrementar_su_planta_de_person': 1},
                  'Bogotá Productiva Local':{'ANTIGUEDAD_MAYOR_UN_AÑO': 1,
                                             'NO_TRABAJADORES_MAYOR_9': 0},
                  'Camino a la inclusión financiera': {'CONTABILIDAD': 'No llevo contabilidad',
                                                       'INTERES_PERFIL':1,
                                                       'CURSO_ED_FINANCIERA':'Si'}, #No son excluyentes, si se cumple alguna puede ser priorizado
                  'Corazon Productivo': {'NO_TRABAJADORES_MAYOR_9': 1,
                                         'CORAZON_PRODUCTIVO': 1,
                                         'CIIU_DOS_DIGITOS_CATEGORIA': 'INDUSTRIAS MANUFACTURERAS',
                                         'TIPO_EMPLAZAMIENTO': 'Local_oficina_fábrica_o_bodega'},
                  'Bogotá Alto Impacto': {'ANTIGUEDAD_MAYOR_UN_AÑO': 1,
                                          'R_MERCANTIL':'Si y la renovó el último año'},
                  'Hecho en Bogotá': {'DIGITAL':'Si',
                                          'INVERTIR':'Inventario',
                                      'ANTIGUEDAD_MAYOR_TRES_AÑOS':1}

                  }


tipo_dict = {'Agencia Pública de Empleo':'No excluyente',
            'Pago por resultados o Empleo Joven':'No excluyente',
            'Bogotá Productiva Local':'Excluyente',
            'Camino a la inclusión financiera': 'No excluyente', #No son excluyentes, si se cumple alguna puede ser priorizado
            'Corazon Productivo': 'Excluyente',
            'Bogotá Alto Impacto': 'Excluyente',
            'Hecho en Bogotá': 'Excluyente'
                  }


#Definir varibles a inlcuir en ejercicio de cluster
features_cont = ['EDAD', 'INGRESOS', 'NO_TRABAJADORES']
features_cat = ['DIGITAL', 'ANTIGUEDAD', 'PROPIETARIOS_MUJERES_dummy', 'PROPIETARIOS_HOMBRES_dummy', 'R_MERCANTIL', 'RAZON_CREAR_NEGOCIO']

features = ['EDAD', 'INGRESOS', 'NO_TRABAJADORES', 'DIGITAL', 'ANTIGUEDAD', 'PROPIETARIOS_MUJERES_dummy', 'PROPIETARIOS_HOMBRES_dummy', 'R_MERCANTIL', 'RAZON_CREAR_NEGOCIO']



