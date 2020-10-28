import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO, StringIO
from PIL import Image
from enum import Enum
from typing import Union
img = Image.open('FP.png')
st.beta_set_page_config(page_title="ForePower", page_icon=img)

st.sidebar.image(img, caption='',use_column_width=True)
st.sidebar.title("VVD Energy Control S.A.S.")
st.sidebar.write('---')
st.sidebar.header("Por favor elija una opción")
Radio_select = st.sidebar.radio("", ('Calculadora pu','Predicción de la demanda','Tutorial y otros'))
df3 = pd.read_csv('Real2020.csv', sep=',')
df4 = pd.read_csv('Loadfhd.csv', sep=',')
df5 = pd.read_csv('Date2020.csv', sep=',')


def to_excel(df5):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df3.to_excel(writer, sheet_name='Loadfhd')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df5):

    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df3)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Loadfhd.xlsx">Descargar datos de la carga </a>' # decode b'abc' => abc

def show_raw_visualization(data):
    time_data = data.index
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 1],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)

    plt.show()

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

def pu():
    st.title("""
    Calculadora por unidad
        """)

    st.write('---')
    st.write("""El programa presentado en esta sección de nuestra aplicación permite realizar calculos en el sistema de *pu* del siguiente esquema,
    en donde el usuario podra ingresar los valores de cada una de las variables que se muestran en el mismo y obtener el valor correspondiente en *pu*,
    es importante recordar que se debe leer atentamente a cada una de las casillas de ingreso de datos para que no se cometan errores. """)
    st.write('---')


    st.sidebar.write('---')
    st.sidebar.title("""
     Ingreso de datos
    """)

    image = Image.open('Sys1.png')
    st.image(image, caption='Diagrama general del circuito', use_column_width=True)

    st.write('---')
    st.write("""Es importante que usted sepa que durante la etapa de incersión de datos, pueden llegar a aparecer diferentes tipos de mensajes,
    ya sea para informarle que posee algún tipo de error o para recordarle que debe seguir ingresando sus datos, también tenga en cuenta que si deja casillas
    vacias, la aplicación le recordara mediante un mensaje de error que debe llenarla adecuadamente""")
    image = Image.open('Sys2.png')
    st.image(image, caption='Diagrama circuital re escrito', use_column_width=True)

    st.sidebar.write('---')
    st.sidebar.header('Bases y generadores  ')
    sbase=float(st.sidebar.number_input("Introduzca la potencia base del sistema en kVA: "))
    vbase1=float(st.sidebar.number_input("Introduzca la tensión base de la sección 1 en kV: "))
    vbase2=float(st.sidebar.number_input("Introduzca la tensión base de la sección 2 en kV: "))
    vbase3=float(st.sidebar.number_input("Introduzca la tensión base de la sección 3 en kV: "))
    vbase4=float(st.sidebar.number_input("Introduzca la tensión base de la sección 4 en kV: "))
    vbase5=float(st.sidebar.number_input("Introduzca la tensión base de la sección 5 en kV: "))
    vgen1=float(st.sidebar.number_input("Cuál es la tensión nominal del primer generador en kV: "))
    sgen1=float(st.sidebar.number_input("Cuál es la potencia nominal del primer generador en kVA: "))
    xpgen1=float(st.sidebar.number_input("Cuál es la reactancia del primer generador (esta es porcentual): "))


    xgenb1=float(0)
    vgen1pu=float(0)

    if  sgen1 <= 0 or vbase1 <= 0:
            st.text("")
    else:
            xgenb1=(sbase/sgen1)*((vgen1/vbase1)**2)*(xpgen1/100)
            vgen1pu=vgen1/vbase1

    namedata1 = {'Variables':['sbase','vbase1' ,'vbase2','vbase3','vbase4','vbase5','vgen1',
                              'sgen1','xpgen1' ,'xgenb1','vgen1pu'],
                 'Valor':[sbase,vbase1,vbase2,vbase3,vbase4,vbase5,vgen1,sgen1,xpgen1,xgenb1,
                          vgen1pu]}
    ###########################################################################


    st.sidebar.header('Trafo 1')
    VaT1 =float(st.sidebar.number_input("Lado de alta T1: "))
    VbT1 = float(st.sidebar.number_input("Lado de baja T1: "))
    xT1=float(st.sidebar.number_input("Cuál es la reactancia del primer transformador (esta es porcentual): "))
    sT1=float(st.sidebar.number_input("Introduzca la potencia en kVA de T1: "))
    a=abs(VaT1-vgen1)
    b=abs(VbT1-vgen1)

    if b<=(0.05*vgen1):
      VaT1, VbT1= VbT1, VaT1
      ##print(b,VaT1,VbT1)

    if (b<=0.05*vgen1 or a<=0.05*vgen1):
        try:
            xpuT1=(sbase/sT1)*((VaT1/vbase1)**2)*(xT1/100)
        except:
            st.success("Sigue introduciendo tus datos")
      ##st.text("El valor pu de la reactancia de T1 es:",xpuT1)
    else:
      st.error("Datos erroneos")


    st.sidebar.header('Lineas')

    zlin1auxre=float(st.sidebar.number_input("cuál es la impedancia de la linea 1, en ohm (Parte Real): "))
    zlin1auxim=float(st.sidebar.number_input("cuál es la impedancia de la linea 1, en ohm (Parte imaginaria): "))
    zlin1 = complex(zlin1auxre + (zlin1auxim*1j))
    try:
        zlin1pu=zlin1/((vbase2**2)/sbase)
    except:
        st.text("")
    ##print("La impedancia en pu de la linea es: ",zlin1pu)

    vgen2=float(st.sidebar.number_input("Cuál es la tensión nominal del segundo generador en kV: "))
    sgen2=float(st.sidebar.number_input("Cuál es la potencia nominal del segundo generador en kVA: "))
    xpgen2=float(st.sidebar.number_input("Cuál es la reactancia del segundo generador (esta es porcentual): "))
    try:
        xgenb3=(sbase/sgen2)*((vgen2/vbase3)*2)*(xpgen2/100)
        vgen2pu=vgen2/vbase3
    except:
        st.text("")
    ##print("la reactancia del generador 1 con las nuevas bases en pu es: ",xgenb3)


    st.sidebar.header("Trafo 2")
    VaT2 = float(st.sidebar.number_input("Lado de alta T2: "))
    VbT2 = float(st.sidebar.number_input("Lado de baja T2:"))
    xT2=float(st.sidebar.number_input("Introduzca la reactancia del segundo transformador (esta es porcentual): "))
    sT2=float(st.sidebar.number_input("Introduzca la potencia en kVA de T2: "))
    ##print(abs((-2)))
    a=abs(VaT2-vgen2)
    b=abs(VbT2-vgen2)
    if a<=(0.05*vgen2):
        VbT2, VaT2= VaT2, VbT2
        ##print(b,VaT2,VbT2)

    if (a<=0.05*vgen2 or b<=0.05*vgen2):
        try:
            xpuT2=(sbase/sT2)*((VbT2/vbase2)**2)*(xT2/100)
        except:
            st.text("")
        ##print("El valor pu de la reactancia de T1 es: ",xpuT2)
    else:
        st.error('Datos Erroneos')
        ##print("Datos erroneos")


    st.sidebar.header("Trafo 3")

    VaT3 = float(st.sidebar.number_input("Lado de alta T3: "))
    VbT3 = float(st.sidebar.number_input("Lado de baja T3: "))
    xT3=float(st.sidebar.number_input("Introduzca la reactancia del tercer transformador (esta es porcentual): "))
    sT3=float(st.sidebar.number_input("Introduzca la potencia en kVA de T3: "))
    a=abs(VaT3-vgen1)
    b=abs(VbT3-vgen1)
    if b<=(0.05*vgen1):
        VaT3, VbT3= VbT3, VaT3
        ##print(b,VaT3,VbT3)

    if (b<=0.05*vgen1 or a<=0.05*vgen1):
        try:
            xpuT3=(sbase/sT3)*((VaT2/vbase1)**2)*(xT3/100)
        except:
            st.text("")
        ##print("El valor pu de la reactancia de T1 es: ",xpuT3)
    else:
        st.error('Datos Erroneos')
        ##print("Datos erroneos")

    zlin2auxre=float(st.sidebar.number_input("cuál es la impedancia de la linea 2, en ohm (Parte Real): "))
    zlin2auxim=float(st.sidebar.number_input("cuál es la impedancia de la linea 2, en ohm (Parte imaginaria): "))
    zlin2 = complex(zlin2auxre + (zlin2auxim*1j))
    try:
        zlin2pu=zlin2/((vbase4**2)/sbase)
    except:
        st.text("")
    ##print("La impedancia en pu de la linea es: ",zlin2pu)
    zlin3auxre=float(st.sidebar.number_input("cuál es la impedancia de la linea 3, en ohm (Parte Real): "))
    zlin3auxim=float(st.sidebar.number_input("cuál es la impedancia de la linea 3, en ohm (Parte imaginaria): "))
    zlin3 = complex(zlin3auxre+(zlin3auxim*1j))
    try:
        zlin3pu=zlin3/((vbase4**2)/sbase)
    except:
        st.text("")
    ##print("La impedancia en pu de la linea 3 es: ",zlin3pu)

    st.sidebar.header("Trafo 4")
    VaT4= float(st.sidebar.number_input("Lado de alta T4:"))
    VbT4= float(st.sidebar.number_input("Lado de baja T4:"))
    xT4=float(st.sidebar.number_input("Introduzca la reactancia del transformador 4(esta es porcentual): "))
    sT4=float(st.sidebar.number_input("Introduzca la potencia en kVA de T4: "))
    a=abs(VaT4-vgen2)
    b=abs(VbT4-vgen2)
    if a<=(0.05*vgen2):
        VbT4, VaT4= VaT4, VbT4
        ##print(a,VaT4,VbT4)

    if (b<=0.05*vgen2 or a<=0.05*vgen2):
        try:
            xpuT4=(sbase/sT4)*((VbT4/vbase3)**2)*(xT4/100)
        except:
            st.text("")
        ##print("El valor pu de la reactancia de T1 es: ",xpuT4)
    else:
        st.error("Datos Erroneos")
        #print("Datos erroneos")

    vgen3=float(st.sidebar.number_input("Cuál es la tensión nominal del tercer generador en kV: "))
    sgen3=float(st.sidebar.number_input("Cuál es la potencia nominal del tercer generador en kVA: "))
    xpgen3=float(st.sidebar.number_input("Cuál es la reactancia del tercer generador (esta es porcentual): "))
    try:
        xgenb5=(sbase/sgen3)*((vgen3/vbase5)**2)*(xpgen3/100)
        vgen3pu=vgen3/vbase5
    except:
        st.text("")
    ##print("la reactancia del generador 3 con las nuevas bases en pu es: ",xgenb5)


    st.sidebar.header("Trafo 5")
    VaT5 = float(st.sidebar.number_input("Lado de alta T5: "))
    VbT5=  float(st.sidebar.number_input("Lado de baja T5: "))
    xT5=float(st.sidebar.number_input("Introduzca la reactancia del transformador 5(esta es porcentual): "))
    sT5=float(st.sidebar.number_input("Introduzca la potencia en kVA de T5: "))
    a=abs(VaT5-vgen3)
    b=abs(VbT5-vgen3)

    if a<=(0.05*vgen3):
        VbT5, VaT5= VaT5, VbT5
        ##print(a,VaT5,VbT5)

    if (b<=0.05*vgen3 or a<=0.05*vgen3):
        try:
            xpuT5=(sbase/sT5)*((VbT5/vbase5)**2)*(xT5/100)
        except:
            st.text("")
        ##print("El valor pu de la reactancia de T1 es: ",xpuT5)
    else:
        st.error("Datos Erroneos")
        ##print("Datos erroneos")

    try:
        xgenb1=xgenb1*1j
        xgenb3=xgenb3*1j
        xgenb5=xgenb5*1j
        z1tot=(xpuT1*1j)+zlin1pu+xpuT2*1j
        z2tot=xpuT3*1j+zlin2pu
        z3tot=zlin3pu+xpuT4*1j
        z4tot=xgenb5*1j+xpuT5*1j
        Va=((vgen1pu*(xgenb3*(z1tot*(z4tot-z2tot)+z3tot*(z4tot-z2tot)+z4tot*z2tot)+z1tot*(z3tot*(z4tot-z2tot)+z4tot*z2tot))*z1tot-(vgen2pu*z1tot*(z3tot*(z4tot-z2tot)+z4tot*(z1tot+z2tot))-vgen3pu*(xgenb3*(z1tot*(z1tot+z2tot)+z3tot*z1tot)+z1tot*z3tot*z1tot))*xgenb1)/(xgenb3*(z1tot*(z4tot*z1tot+xgenb1*(z1tot+z2tot)-z1tot*z2tot)+(z3tot*(z4tot+xgenb1-z2tot)+z4tot*z2tot)*z1tot)-z1tot*(z3tot*(z4tot*(xgenb1-z1tot)-xgenb1*(z1tot+z2tot)+z1tot*z2tot)+z4tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot))))
        Vb=((vgen1pu*xgenb3*(z1tot*z4tot+z3tot*(z4tot-z2tot)+z4tot*z2tot)*z1tot-vgen2pu*z1tot*(z3tot*(z4tot*(xgenb1-z1tot)-xgenb1*(z1tot+z2tot)+z1tot*z2tot)+z4tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot))+vgen3pu*xgenb3*(z1tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot)+z3tot*xgenb1*z1tot))/(xgenb3*(z1tot*(z4tot*z1tot+xgenb1*(z1tot+z2tot)-z1tot*z2tot)+(z3tot*(z4tot+xgenb1-z2tot)+z4tot*z2tot)*z1tot)-z1tot*(z3tot*(z4tot*(xgenb1-z1tot)-xgenb1*(z1tot+z2tot)+z1tot*z2tot)+z4tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot))))
        Vc=((vgen1pu*(xgenb3*(z1tot+z3tot+z2tot)+z1tot*z3tot)*z4tot*z1tot-vgen2pu*z1tot*(z3tot*xgenb1+xgenb1*(z1tot+z2tot)-z1tot*z2tot)*z4tot+vgen3pu*(xgenb3*(z1tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot)+z3tot*(xgenb1-z2tot)*z1tot)+z1tot*z3tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot)))/(xgenb3*(z1tot*(z4tot*z1tot+xgenb1*(z1tot+z2tot)-z1tot*z2tot)+(z3tot*(z4tot+xgenb1-z2tot)+z4tot*z2tot)*z1tot)-z1tot*(z3tot*(z4tot*(xgenb1-z1tot)-xgenb1*(z1tot+z2tot)+z1tot*z2tot)+z4tot*(xgenb1*(z1tot+z2tot)-z1tot*z2tot))))
        I1pu=((vgen1pu-Va)/xgenb1)/(3**(1/2))
        I2pu=((Va-Vb)/z1tot)/(3**(1/2))
        I3pu=((Va-Vc)/z2tot)/(3**(1/2))
        I4pu=((Vc-Vb)/z3tot)/(3**(1/2))
        I5pu=((vgen2pu-Vc)/z4tot)/(3**(1/2))
    except:
        st.text("")
        ##print("Va es: ",Va)
        ##print("Vb es: ",Vb)
        #print("Vc es: ",Vc)

    try:
        namedata2 = {'Variables':['Reactancia por unidad generador 1','Tensión en pu del generador 1', 'Valor pu de la reactancia de T1',
                              'Impedancia en pu de la linea 1','Reactancia del generador 1, nuevas base', 'Valor pu de la reactancia de T2',
                              'Valor pu de la reactancia de T3', 'Impedancia en pu de la linea 2','Impedancia en pu de la linea 3','Valor pu de la reactancia de T4',
                              'Reactancia del generador 3,nuevas bases','Valor pu de la reactancia de T5','Va es','Vb es','Vc es', 'I1 es','I2 es' ,'I3 es' ,'I4 es' ,'I5 es'],
                 'Valor': [xgenb1 ,vgen1pu, xpuT1, zlin1pu,xgenb3,xpuT2,xpuT3,zlin2pu,zlin3pu,xpuT4,xgenb5,xpuT5,Va,Vb,Vc,I1pu,I2pu,I3pu,I4pu,I5pu]}
        features = pd.DataFrame(namedata1)
        features2 = pd.DataFrame(namedata2)
        return features, features2
    except:
        st.error("Algo anda mal, revisa tus datos")

def predict():

    st.title("Predicción de la demanda energética")
    st.write("---")
    st.write("""En esta sección de la aplicación, encontrarás todo lo relacionado a la predicción de la demanda realizada por medio de redes neuronales,
    para la primera semana del 2020 en el sector del valle, de EMIC (Empresas municipales de Cali), a continuación, encontrarás los pasos a seguir si
    quieres realizar la predicción con tus propios datos y los resultados obtenidos de este ejercicio. Lo primero que debes saber, es que este modelo de predicción únicamente tiene en cuenta tres variables,
     la fecha de la toma de la medida, la carga y la temperatura, sin embargo, con estos datos es suficiente para realizar una predicción bastante acertada de la demanda de energía.""")
    st.write("---")
    st.header("Prepación de datos")
    st.write("""Esta primera imagen es la visualización de la carga y la temperatura de un día del año 2019 en el sitio en el cuál se realizó la predicción. """)
    image = Image.open('Actual Day Load 2019.png')
    st.image(image, caption='Actual Day Load 2019', use_column_width=True)
    st.write("""En esta grafica se visualiza la carga y la temperatura de todo el año 2019, detallando su comportamiento""")
    image = Image.open('Actual January Load 2019.png')
    st.image(image, caption='Diagrama general del circuito', use_column_width=True)
    st.write("""En esta imagen se aprecia la misma distribución de datos que las imágenes anteriores, pero para la primera semana del año 2019 """)
    image = Image.open('Actual Week Load 2019.png')
    st.image(image, caption='Actual Week Load 2019', use_column_width=True)
    st.write("""Esta imagen representa la correlación que se da por medio de la carga y la temperatura, como se puede apreciar, en esta zona especifica no influye de una manera decisoria, la temperatura como tal.""")
    image = Image.open('MapadeCorrelacion.png')
    st.image(image, caption='Mapa de Correlacion', use_column_width=True)
    st.write("""Por último, se aprecian los márgenes de error en donde el eje X indica la cantidad de veces que se reentreno la red hasta llegar a un margen de error del 7%.""")
    image = Image.open('TrainingAndValidationLoss.png')
    st.image(image, caption='Training And Validation Loss', use_column_width=True)
    st.write("---")
    st.write("""Luego de preparar los datos que se utilizaron para realizar este ejercicio, se realizó un script en base a Keras que permite la predicción de datos mediante un entrenamiento anterior.
    A raíz de estos datos y de realizar el procesamiento de los mismos, se obtiene la siguiente predicción
    """)
    st.write("---")
    st.header("Resultados de la predicción")
    st.write("""En esta primera grafica se detalla la predicción hecha para la primera semana del año 2020, en donde si se compara con la predicción del año 2019, se puede apreciar una similitud""")
    image = Image.open('prediccion.png')
    st.image(image, caption='Prediction load', use_column_width=True)
    st.write("""Esta grafica detalla los datos usados para el entrenamiento de la red neuronal, la carga de todo el año 2019 y la predicción """)
    image = Image.open('PredictionAndActualLoad.png')
    st.image(image, caption='Prediction And Actual Load', use_column_width=True)
    st.write("""En este grafico se aprecia la predicción obtenida de la primera semana del 2020 y de todo el mes de diciembre del año 2019, se aprecia un patrón de carga """)
    image = Image.open('predictionAndDecember.png')
    st.image(image, caption='Prediction And December', use_column_width=True)
    st.write("""En este grafico se aprecia la comparación de la curva real vs la predicción del día 3 de enero de 2020""")
    image = Image.open('Realvspredicted3January.png')
    st.image(image, caption='Real vs predicted 3 January', use_column_width=True)
    st.write("""Por ultimo se muestra la predicción vs la demanda real, para toda la primera semana del año 2020""")
    image = Image.open('prediccionVsReal.png')
    st.image(image, caption='Prediction Vs Real first week', use_column_width=True)
    st.write("---")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        datospredic = pd.read_csv(uploaded_file)
        st.dataframe(datospredic)
        if st.button("Correr programa de predicción"):
            #Insert .csv file
            df = datospredic

            #Function to plot
            titles = ["Load","Temperature"]
            feature_keys = ["Load", "Temperature"]
            colors = ["blue","orange"]

            #First data view

            #plot All data
            show_raw_visualization(df)
            plt.savefig('Actual Year Load.png')

            #plot Day data
            show_raw_visualization(df.iloc[0:24,:])
            plt.savefig('Actual Day Load.png')

            #plot Week data
            show_raw_visualization(df.iloc[0:24*7,:])
            plt.savefig('Actual Week Load.png')

            #plot Month January data
            show_raw_visualization(df.iloc[0:24*7*31,:])
            plt.savefig('Actual January Load.png')


            #Correlation plot

            show_heatmap(df)

            plt.savefig('CorrelationHeatmap.png')


            # Select features (columns) to be involved intro training and predictions
            cols = list(df)[0:2]

            # Extract dates (will be used in visualization)
            datelist_train = list(df.index)
            print('Training set shape == {}'.format(df.shape))
            print('All timestamps == {}'.format(len(df.index)))
            print('Featured selected: {}'.format(cols))

            dataset_train = df
            dataset_train = dataset_train[cols].astype(str)
            for i in cols:
                for j in range(0, len(dataset_train)):
                    dataset_train[i][j] = dataset_train[i][j].replace(',', '')

            dataset_train = dataset_train.astype(float)

            # Using multiple features (predictors)
            training_set = dataset_train.values

            print('Shape of training set == {}.'.format(training_set.shape))
            training_set

            # Feature Scaling
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler()
            training_set_scaled = sc.fit_transform(training_set)

            sc_predict = StandardScaler()
            sc_predict.fit_transform(training_set[:, 0:1])

            # Creating a data structure with 90 timestamps and 1 output
            X_train = []
            y_train = []

            #Default predicted_days
            predicted_days = 7
            """ INSERT LINE TO INSERT NUMBER OF DAYS TO PREDICT """
            n_future = 24*predicted_days   # Number of days we want top predict into the future
            n_past = 24*31     # Number of past days we want to use to predict the future


            for i in range(n_past, len(training_set_scaled) - n_future +1):
                X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
                y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)

            print('X_train shape == {}.'.format(X_train.shape))
            print('y_train shape == {}.'.format(y_train.shape))

            """#2. Create Model
            ## Training: Building the LSTM based Neural Network
            """

            # Import Libraries and packages from Keras
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            from keras.layers import Dropout
            from keras.optimizers import Adam

            # Initializing the Neural Network based on LSTM
            model = Sequential()

            # Adding 1st LSTM layer
            model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))

            # Adding 2nd LSTM layer
            model.add(LSTM(units=10, return_sequences=False))

            # Adding Dropout
            model.add(Dropout(0.25))

            # Output layer
            model.add(Dense(units=1, activation='linear'))

            # Compiling the Neural Network
            model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

            """## Start Training"""

            visualize_loss(history, "Training and Validation Loss")
            plt.savefig('Training and Validation Loss.png')


            """#3. Make Future predictions"""

            # Generate list of sequence of days for predictions
            datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

            # Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
            #I have to create a new datelist_future

            datelist_future_ = []
            for this_timestamp in datelist_future:
                datelist_future_.append(this_timestamp.date())

            df2020 = pd.read_csv('Date2020.csv',index_col = 0, parse_dates = [0])

            dfReal2020 = pd.read_csv('Real2020.csv',index_col = 0, parse_dates = [0])

            datelist_future_hour= list(df2020.index[0:168])
            type(datelist_future_hour), type(datelist_future)
            #type(dfReal2020)

            """## 5. Make predictions for future dates"""

            # Perform predictions
            predictions_future = model.predict(X_train[-n_future:])
            predictions_train = model.predict(X_train[n_past:])

            # Inverse the predictions to original measurements

            # ---> Special function: convert <datetime.date> to <Timestamp>
            y_pred_future = sc_predict.inverse_transform(predictions_future)
            y_pred_train = sc_predict.inverse_transform(predictions_train)


            PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Load']).set_index(pd.Series(datelist_future_hour))
            PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Load']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

            # Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
            PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

            """## 6. Visualize data

            ###Whole Data
            """

            # Set plot size
            from pylab import rcParams
            rcParams['figure.figsize'] = 20,8

            # Plot parameters
            START_DATE_FOR_PLOTTING = '2019-01-01'

            plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
            plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
            plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')

            plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

            plt.grid(which='major', color='#cccccc', alpha=0.5)

            plt.legend(shadow=True)
            plt.title('Predictions and Actual Load', family='Arial', fontsize=12)
            plt.xlabel('Timeline', family='Arial', fontsize=10)
            plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.show()
            plt.savefig('Actual vs predicted load.jpg')


            """### Just Prediction"""

            rcParams['figure.figsize'] = 20, 8

            # Plot parameters
            START_DATE_FOR_PLOTTING = '2020-01-01'

            plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
            #plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
            #plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
            plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

            plt.grid(which='major', color='#cccccc', alpha=0.5)

            plt.legend(shadow=True)
            plt.title('Predictions Load', family='Arial', fontsize=12)
            plt.xlabel('Timeline', family='Arial', fontsize=10)
            plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.show()
            plt.savefig('Predicted Load.jpg')


            """### December 2019 + 2020"""

            rcParams['figure.figsize'] = 20, 8

            # Plot parameters
            START_DATE_FOR_PLOTTING = '2019-12-01'

            plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
            #plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
            plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
            plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

            plt.grid(which='major', color='#cccccc', alpha=0.5)

            plt.legend(shadow=True)
            plt.title('Predictions Load', family='Arial', fontsize=12)
            plt.xlabel('Timeline', family='Arial', fontsize=10)
            plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.show()
            plt.savefig('December2019andJanuary2020.jpg')


            #Compare real with prediction

            rcParams['figure.figsize'] = 20, 8

            # Plot parameters
            START_DATE_FOR_PLOTTING = '2020-01-01'

            plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Load'], color='r', label='Predicted Load')
            plt.plot(dfReal2020.index[:168],dfReal2020['Load'][:168],color='orange',label ='Real 2020 Load' )
            #plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
            #plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
            plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

            plt.grid(which='major', color='#cccccc', alpha=0.5)

            plt.legend(shadow=True)
            plt.title('Predictions Load', family='Arial', fontsize=12)
            plt.xlabel('Timeline', family='Arial', fontsize=10)
            plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.show()
            plt.savefig('Realvspredicted.png')


            #Compare real with prediction

            rcParams['figure.figsize'] = 20, 8

            # Plot parameters
            START_DATE_FOR_PLOTTING = '2020-01-03'

            plt.plot(PREDICTIONS_FUTURE.index[48:73], PREDICTIONS_FUTURE['Load'][48:73], color='r', label='Predicted Load')
            plt.plot(dfReal2020.index[48:73],dfReal2020['Load'][48:73],color='orange',label ='Real 2020 Load' )
            #plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Load'], color='orange', label='Training predictions')
            #plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Load'], color='b', label='Actual Load')
            plt.axvline(x = min(PREDICTIONS_FUTURE.index[48:73]), color='green', linewidth=2, linestyle='--')

            plt.grid(which='major', color='#cccccc', alpha=0.5)

            plt.legend(shadow=True)
            plt.title('Predictions Load', family='Arial', fontsize=12)
            plt.xlabel('Timeline', family='Arial', fontsize=10)
            plt.ylabel('Energy (MW)', family='Arial', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.show()
            plt.savefig('Realvspredicted3January.png')


            """## Preparing forecasting.csv"""

            forecasting = PREDICTIONS_FUTURE
            forecasting.to_csv('forecasting.csv')


    to_excel(df5)
    get_table_download_link(df5)
    st.markdown(get_table_download_link(df5), unsafe_allow_html=True)





if Radio_select =='Calculadora pu':
    try:
        df1,df2 = pu()
        st.header("Datos ingresados")
        st.write("""En esta sección encontrará los datos ingresados por el usuario en una primera instancia,
        dando a conecer todas las variables ingresadas, ya sea para una posible revisión o corrección que cometa el usuario""")
        st.table(df1)
        st.header("Resultados")
        st.write("""En esta sección usted encontrará los resultados de los calculos correspondientes que se realizaran a partir de los valores
        ingresados en la sección anterior, todas las tablas se mostrarán de manera detallada y usted podrá ver sus resultados con el nombre de las
        variables""")
        st.table(df2.astype('object'))
        st.write('---')
        st.write("""Por último, usted contará con la opción de descargar los resultados y los datos ingresados en dos archivos de extensión .csv, en donde podra consultar los datos
        que usted visualiza en las dos tablas de las secciones anteriores""")
        if st.button("Descarga tus datos y resultados"):
            df1.to_csv('Datos ingresados.csv', header=False, index=False)
            df2.to_csv('Resultados.csv', header=False, index=False)
        else:
            st.text("")
    except:
        st.text("")

elif Radio_select =='Predicción de la demanda' :
    predict()
else:
    st.title("Tutorial/Guia de usuario")
    st.write("""Dentro de este apartado de la aplicación, encontrarás un video instructivos
    acerca del funcionamiento de nuestra aplicación y de todas sus funcionalidades""")
    video_file = open('vid.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.header("Contacto y sugerencias")
    st.write("""Esta aplicación fue diseñada por el equipo de trabajo correspondiente a Valentina Alvarez Villa, Victor Daniel Caicedo Garcia, Daniel Alejandro Teran Fernandez """)
    st.header("Correos de contacto")
    st.write("vdcaicedog@unal.edu.co")
    st.write("valvarezv@unal.edu.co")
    st.write("dteranf@unal.edu.co")
