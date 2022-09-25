from select import select
import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from streamlit_option_menu import option_menu
from matplotlib.pyplot import figure
from PIL import Image
from st_aggrid import AgGrid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, plot_confusion_matrix



st.set_page_config(page_icon="🚀", page_title="DDOS", initial_sidebar_state="auto")

#Menu Navigasi
pilih = option_menu(
  menu_icon="diagram-3",
  menu_title="Identifikasi DDOS Random Forest",
  options=["Beranda", "Visual", "Tentang"],
  icons=["house", "graph-up", 'file-earmark-person'],
  default_index=0,
  orientation="horizontal"
)

if pilih == "Beranda":
  st.sidebar.header("Input Parameter")
  df = pd.read_csv('dataset_sdn.csv')
  data = st.sidebar.checkbox("Dataset")
  if data:
    st.subheader("Dataset")
    st.dataframe(df)
  
  numeric_df = df.select_dtypes(include=['int64', 'float64'])
  object_df = df.select_dtypes(include=['object'])
  numeric_cols = numeric_df.columns
  object_cols = object_df.columns
  
  to_drop = ['dt']
  df = df.drop(to_drop, axis='columns')
  df = df[df['pktrate'] !=0]

  df['src'] = [int(i.split('.')[3]) for i in df['src']]
  df['dst'] = [int(i.split('.')[3])for i in df['dst']]
  df['switch'] = df['switch'].astype(str) 
  df['src'] = df['src'].astype(str)
  df['dst'] = df['dst'].astype(str)
  df['port_no'] = df['port_no'].astype(str)
  df['Protocol'] = df['Protocol'].astype(str)

  new_df = pd.get_dummies(df,columns = ['switch','src','Protocol','dst','port_no'])

  
  new_df.fillna(new_df.mean(), inplace=True)
  x=new_df.drop(['label'],axis=1)
  y=new_df.label
  x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.25)

  training = st.sidebar.checkbox("Data Training dan Testing")
  if training:
    st.subheader("Data Training(75%) dan Testing (25%)")
    AgGrid(new_df.head(10))

  
  
  st.sidebar.subheader("Random Forest Classifier")
  maxdepth = st.sidebar.slider('max_depth (Kedalaman)', value=2, max_value=10, min_value=0)
  maxfitur = st.sidebar.slider("max_features (Fitur)", value=16, max_value=22, min_value=0)
  estimator = st.sidebar.slider("n_estimator (Jumlah Pohon)", value=200, min_value=0, max_value=300)
  random_state = st.sidebar.slider("random_state (Keadaan Acak)", max_value=18, min_value=None, value=5)


  model = RandomForestClassifier(n_estimators=(estimator), max_depth=(maxdepth), random_state=(random_state), max_features=(maxfitur), bootstrap=True)
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  
  perhitungan = st.sidebar.checkbox("Nilai Perhitungan")
  col3, col4 = st.columns(2)
  if perhitungan:
    with col3:
      st.metric("Nilai Akurasi", accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), help="rasio prediksi Benar (positif dan negatif) dengan keseluruhan data")
    with col4:
      st.metric("F1 Skor", f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), help=" perbandingan rata-rata presisi dan recall yang dibobotkan")

  menu_opsi = st.sidebar.multiselect("Hasil Klasifikasi", ['Classification Report', 'Confusion Matrix', 'Plot Confusion Matrix'])
  col1, col2 = st.columns(2)
  if 'Confusion Matrix' in menu_opsi:
    with col2:
      st.subheader("Confusion Matrix")
      st.table(confusion_matrix(y_test, y_pred))
  
  if 'Classification Report' in menu_opsi:
    with col1:
      st.subheader("Classification Report")
      st.text(classification_report(y_test, y_pred))

  if 'Plot Confusion Matrix' in menu_opsi:
    st.subheader("Plot Confusion Matrix")
    class_names = ['0', '1']
    plot_confusion_matrix(model, x_test, y_test, display_labels=class_names, cmap=plt.cm.Blues)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


elif pilih == "Visual":
  df = pd.read_csv("dataset_sdn.csv")

  numeric_df = df.select_dtypes(include=['int64', 'float64'])
  object_df = df.select_dtypes(include=['object'])
  numeric_cols = numeric_df.columns
  object_cols = object_df.columns

  to_drop = ['dt']
  df = df.drop(to_drop, axis='columns')
  df = df[df['pktrate'] !=0]

  df['src'] = [int(i.split('.')[3]) for i in df['src']]
  df['dst'] = [int(i.split('.')[3])for i in df['dst']]
  df['switch'] = df['switch'].astype(str) 
  df['src'] = df['src'].astype(str)
  df['dst'] = df['dst'].astype(str)
  df['port_no'] = df['port_no'].astype(str)
  df['Protocol'] = df['Protocol'].astype(str)

  new_df = pd.get_dummies(df,columns = ['switch','src','Protocol','dst','port_no'])

  new_df.fillna(new_df.mean(), inplace=True)
  x=new_df.drop(['label'],axis=1)
  y=new_df.label
  x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.25)

  fig6 = plt.figure()
  sns.countplot(df[df.label == 1].Protocol, palette="magma")
  plt.title('Jumlah Serangan Berbahaya dari protokol yang berbeda')

  fig7 = plt.figure()
  sns.countplot(df.Protocol, palette="magma")
  plt.title('Jumlah Serangan Tidak Berbahaya dari protokol yang berbeda')

  fig8 = plt.figure()
  sns.countplot(df.switch, palette="rainbow")
  plt.title('Jumlah Seluruh Switch yang terpakai  ')

  fig1 = plt.figure()
  plt.title('Perbandingan Count antara Serangan Tidak Berbahaya dan Berbahaya')
  sns.countplot(df.label)
  
  fig2 = px.bar(list(dict(df.src.value_counts()).keys()), dict(df.src.value_counts()).values(),
  labels={"x": "Jumlah Permintaan", "index": "Jumlah IP Pengirim"},
  template="seaborn", title="Jumlah Semua Serangan Tidak Berbahaya")
  
  fig3 = px.bar(list(dict(df[df.label == 1].src.value_counts()).keys()), dict(df[df.label == 1].src.value_counts()).values(),
  labels={"x": "Jumlah Permintaan", "index": "Jumlah IP Pengirim"},
  template="ggplot2", title="Jumlah Semua Serangan Berbahaya")
  
  fig4 = figure(figsize=(12, 7), dpi=80)
  plt.barh(list(dict(df.src.value_counts()).keys()), dict(df.src.value_counts()).values(), color='#277BC0')
  plt.barh(list(dict(df[df.label == 1].src.value_counts()).keys()), dict(df[df.label == 1].src.value_counts()).values(), color='#D2001A')

  for idx, val in enumerate(dict(df.src.value_counts()).values()):
      plt.text(x = val, y = idx-0.2, s = str(val), color='black', size = 13)

  for idx, val in enumerate(dict(df[df.label == 1].src.value_counts()).values()):
      plt.text(x = val, y = idx-0.2, s = str(val), color='w', size = 13)

  plt.xlabel('Jumlah Permintaan')
  plt.ylabel('Alamat IP Pengirim')
  plt.legend(['Tidak Berbahaya','Berbahaya'])
  plt.title('Jumlah Semua Serangan dari alamat IP yang berbeda')
  
  fig5 = figure(figsize=(10, 6), dpi=80)
  plt.bar(list(dict(df.Protocol.value_counts()).keys()), dict(df.Protocol.value_counts()).values(), color='#277BC0')
  plt.bar(list(dict(df[df.label == 1].Protocol.value_counts()).keys()), dict(df[df.label == 1].Protocol.value_counts()).values(), color='#D2001A')

  plt.text(x = 0 - 0.15, y = 31190 + 200, s = str(31190), color='black', size=17)
  plt.text(x = 1 - 0.15, y = 27527 + 200, s = str(27527), color='black', size=17)
  plt.text(x = 2 - 0.15, y = 17292 + 200, s = str(17292), color='black', size=17)

  plt.text(x = 0 - 0.15, y = 15339 + 200, s = str(15339), color='w', size=17)
  plt.text(x = 1 - 0.15, y = 12474 + 200, s = str(12474), color='w', size=17)
  plt.text(x = 2 - 0.15, y = 7712 + 200, s = str(7712), color='w', size=17)

  plt.xlabel('Protokol')
  plt.ylabel('Count')
  plt.legend(['Tidak Berbahaya', 'Berbahaya'])
  plt.title('Jumlah Semua Serangan dari protokol yang berbeda')
  
  
  labels = ["Tidak Berbahaya", "Berbahaya"]
  sizes = [dict(df.label.value_counts())[0], dict(df.label.value_counts())[1]]
  fig9 = plt.figure(figsize = (13,8))
  plt.pie(sizes, labels=labels, autopct='%1.1f%%',
          shadow=False, startangle=90)
  plt.legend(["Tidak Berbahaya", "Berbahaya"])
  plt.title('Persentasi Data Serangan')

  # Keterangan
  st.sidebar.subheader("Keterangan Graph/Plot :")
  st.sidebar.write("A → Perbandingan Jumlah Count")
  st.sidebar.write("B → Persentasi Data Serangan")
  st.sidebar.write("C → Jumlah Semua Permintaan")
  st.sidebar.write("D → Jumlah Semua Serangan")
  st.sidebar.write("E → Jumlah Permintaan dari Alamat IP berbeda")
  st.sidebar.write("F → Jumlah Permintaan dari Protokol yang berbeda")
  st.sidebar.write("G → Jumlah Serangan Normal dari protokol yang berbeda")
  st.sidebar.write("H → Jumlah Seluruh Switch yang terpakai")
  st.sidebar.write("I → Jumlah Serangan Berbahaya dari protokol yang berbeda")
  
  #Menu Pilihan 
  st.sidebar.subheader("Input Plot/Graph")
  select_option = st.sidebar.multiselect("Pilih Plot/Graph", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
  g1, g2 = st.columns(2)
  g3, g4 = st.columns(2)
  g5, g6, g7 = st.columns(3)
  g8, g9 = st.columns(2)
  if 'A' in select_option:
    with g1:
      fig1
  if 'B' in select_option:
    with g2:
      fig9
  if 'C' in select_option:
    with g9:
      fig2
  if 'D' in select_option:
    with g8:
      fig3
  if 'E' in select_option:
    with g3:
      fig4
  if 'F' in select_option:
    with g4:
      fig5
  if 'G' in select_option:
    with g5:
      fig6
  if 'H' in select_option:
    with g6:
      fig8
  if 'I' in select_option:
    with g7 :
      fig7


elif pilih == "Tentang":
  st.title("Referensi")
  gambar1 = Image.open(r'kaggle.png')
  with st.container():
    image_col, text_col = st.columns((1, 3))
    with image_col:
      st.image(gambar1, caption='Kaggle')
    with text_col:
      st.markdown(""" ### DDOS SDN Dataset
      """)    
      st.markdown("Dataset is called <b>DDOS SDN DATASET</b>. There are 104345 and 23 columns. There is one target called label: contains only 1(malicious) 0(benign)[Lanjutkan Membaca... ](https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset)",
      unsafe_allow_html=True)
  
  gambar2 = Image.open(r'mendeley.png')
  with st.container():
    image2_col, text2_col = st.columns((1,3))
    with image2_col:
      st.image(gambar2, caption='Mendeley Data')
    with text2_col:
      st.markdown(""" ### DDOS attack SDN Dataset
      """)
      st.markdown("This is a SDN specific data set generated by using mininet emulator and used for traffic classification by machine learning and deep learning algorithms.  The project start by creating ten topologies in mininet in which switches are connected to single Ryu controller.[Lanjutkan Membaca...](https://data.mendeley.com/datasets/jxpfjc64kr/1)")