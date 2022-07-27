# Commandes pour lancer le fichier

# cd C:\Users\HP\Documents\DATA\Projet MPG\STREAMLIT\_Dossier final
# streamlit run Streamlit_MPG.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import time

# Partie Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Pour évaluer nos modèles
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.metrics import classification_report_imbalanced

# Eviter les messages d'erreur sur certains graphiques
st.set_option('deprecation.showPyplotGlobalUse', False)

###########################

# Importer glob pour gérer les images
import glob

# Récupérer la liste des maillots
l=glob.glob("media/maillots/*.png")
path_maillots="media/maillots/"

##########################
# Config favicon and title page

st.set_page_config(
        page_title="Py Best Team",
        page_icon="⚽",
        layout="centered",
    )
##########################

header = st.container()
intro = st.container()
dataset = st.container()
dataviz = st.container()
model_training = st.container()
tools = st.container()
final = st.container()

df = pd.read_csv('data/MPG_Current_5j.csv', sep=';') # DF pour la partie Dataviz
df0 = df
df_ml = pd.read_csv('data/MPG_df_final_ML.csv', sep=',', index_col='Unnamed: 0') # DF pour la partie Machine learning

# Création des DF avec les notes différentes de 0
df_00      = df[df['Note'] != 0].sort_values('Journée', ascending=True) # DF global sans les notes moyennes à 0 (joueurs qui ne jouent jamais)
df_00_jour = df[df['Note_Jour_Next']!=0].sort_values('Journée', ascending=True) # DF sans les notes journées à 0

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 0/ HEADER

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

with header:
	st.markdown("<h1 style='text-align: center; color: grey;'>&#9917 Projet <span style='color: #00B727;'>PY BEST TEAM</span> &#9917</h1>", unsafe_allow_html=True)

# Menu horizontal
	choice = option_menu(menu_title = None,
						options = ["Intro", "Dataset", "Dataviz", "Machine learning", "Outils MPG"],
						icons = ["caret-right-square-fill", "server", "file-bar-graph-fill", "fan", "trophy-fill"],
						menu_icon = "bookmark-star",
						default_index = 0,
						orientation = "horizontal",
						styles={
        				"container": {"padding": "500", "background-color": "#EBEBEB"},
        				"icon": {"color": "00B727", "font-size": "16px"}, 
        				"nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#E3FFE5"},
        				"nav-link-selected": {"background-color": "#00B727"}}
						)

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 0/ INTRO

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

if choice == "Intro":
	with intro:
		col1, col2 = st.columns([1,2])
		with col1:
		    st.image('media/logo_mpg.png')
		with col2:
		    st.write("Ce projet est basé sur l'application de ''football fantasy'', **MonPetitGazon**. Elle permet de faire son mercato (choisir ses joueurs en début de saison), et de faire sa composition avant chaque journée. C'est sur ce dernier point que va se baser le reste du projet.")

		st.markdown("<h5 style='text-align: center; color: black;'>&#x1F3AF; <u>OBJECTIF :</u><br><br> Proposer des outils afin de choisir la meilleure équipe pour affronter votre adversaire au prochain match</h5>", unsafe_allow_html=True)

		st.markdown("<h5 style='text-align: center; color: black;'>&#x2B50; <u>LA TEAM</h5>", unsafe_allow_html=True)

		col1, col2, col3, col4, col5 = st.columns(5)
		with col1:
		    st.write("")
		with col2:
		    st.image('media/maillot_benard.png')
		    st.write("[Gayelord BENARD](https://www.linkedin.com/in/gayelord-benard-86918b71/)")
		with col3:
		    st.write("")
		with col4:
		    st.image("media/maillot_castel.png")
		    st.write("[Julien CASTEL](https://www.linkedin.com/in/casteljulien/)")
		with col5:
		    st.write("")

		st.success("A vous la victoire !")

		col1, col2, col3 = st.columns(3)
		with col2:
		    st.image("https://media.giphy.com/media/XcNVCEu7wrAZFciK2P/giphy.gif")

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 1/ DATASET

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

elif choice == "Dataset":

	with dataset:
            st.header('1/ Récupération et préparation des données')
  
            st.markdown("#### :black_medium_small_square: Quel est l'objectif de cette partie ?")
              
            
            st.write("L’objectif principal est de proposer aux joueurs du jeu [MonPetitGazon](https://mpg.football/), des outils leur permettant de mieux analyser les performances des joueurs de Ligue1, dans le but de créer la meilleure composition d’équipe possible pour le prochain match.") 
            st.write("S’agissant d’un jeu thématique reposant sur des résultats réels (le jeu est basé sur les résultats des matchs à l'issue des journées de championnat), **_déterminer les sources utiles et accéder à ces données_** représentent la première étape du projet.")  
  
            st.markdown("#### :black_medium_small_square: Découverte du contexte") 
            
            st.write("Afin de mieux cerner notre sujet, nous avons procédé à quelques recherches et prises de contact via le réseau d'alumni de DataScientest. Nous avons pu notamment interviewé Paul-André Woisard, qui avait rédigé un [article](https://datascientest.com/mpg-le-foot-dans-toute-sa-data) sur le sujet.")         
                        
            st.write ("Nous avons aussi contacter l'équipe de [MPG Stats](https://www.mpgstats.fr/), principale source d'information et de statistiques grand public sur l'univers de MPG.")

            
            st.markdown("#### :black_medium_small_square: Sources de données utilisées")
            st.write("On distingue 2 types de sources d’informations :")
            st.write("* celles directement issue du jeu , via [MPG Stats](https://www.mpgstats.fr/)")
            st.write("* celles venant du contexte thématique , en l'occurrence les résultats des matchs, mesurés via les classements, résumés des matchs, statistiques collectées durant la partie, fiche de club,...")
            st.write("Nous avons opté pour des données exclusivement gratuites et qui auraient été facilement accessibles à un utilisateur de MPG.")

            st.markdown("#### :black_medium_small_square: Travail individuel de chacun des fichiers obtenus")
            
            st.markdown("##### 1/ _MPG_1_ : Export csv via la fonctionnalité native du site MPGstats.")
            st.write("Données ne nécessitant que très peu de retraitement mais usage limité car il est impossible d’en retirer les informations pour chaque match.") 

            st.markdown("##### 2/ _MPG_2_ : Export brut via web scrapping depuis les tableaux du site MPGstats.") 
            st.write("Données nécessitant des retraitements car informations certes disponibles pour chaque match, mais codifiées via l’emploi de certaines chaînes de caractères spéciaux.")

            st.markdown("##### 3/ _Calendrier_ : Export manuel depuis le site [Maxifoot.fr](https://www.maxifoot.fr/)")
            st.write("Export via Datamining simplifié (Interface Power query) afin de récupérer les informations de journées à venir, incluant les dates, et les équipes alignées.")

            st.markdown("##### Voici un aperçu des traitements effectués :")

            st.image("media/Sources_MPG.PNG")

            
            st.write("Pour chacune des sources, les notions importantes permettant des rapprochement ont été identifiées : via les concepts de JOUEUR, JOURNEE et d'EQUIPE. Au niveau des informations collectées, nous avons cherché à déterminer le contexte des notes MPG au niveau de l'équipe et du joueur, ainsi que des métriques MPG comme le nombre de buts, la côte du joueur ou la note donnée par MPG.")

            st.markdown("#### :black_medium_small_square: Mutualisation et préparation du dataframe final")

            st.markdown("##### 1/ _MPG_1_ :")
            st.write("Export initial propre, retirer les colonnes inutiles (non porteuses d’information exploitables pour la suite du projet) , préparation des variables pour rapprocher (“merger”) les sources.") 

            st.markdown("##### 2/ _MPG_2_ :") 
            st.write("Export brut, retirer les colonnes inutiles (colonnes redondantes avec MPG_1), dé-pivoter les colonnes correspondantes aux journées, déduire de l’information claire (nombre de buts dans le match, note du jour, cartons, blessure,...) à partir d’une information codifiée (“décryptage” de la chaîne de caractères), préparation des variables pour rapprocher (“merger”) les sources.")

            st.markdown("##### 3/ _Calendrier_ :")
            st.write("Retraitement effectué via Powerquery, préparation des variables pour rapprocher (“merger”) les sources.")



################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 2/ DATAVIZ

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################


elif choice == "Dataviz":
	with dataviz:
		st.header('2/ **Visualisation des données**')

		st.write("L'objectif de cette section est de présenter divers graphiques réprésentant la distribution de certaines variables clés et les relations qu'elles entretiennent entre elles.")

		show = st.checkbox("<< Cochez cette case si vous souhaitez afficher l'extrait du dataset final")
		if show:
			st.markdown("###### Voici l'extrait du dataframe final que nous allons utiliser pour la visualisation des données.")
			st.write(df.head())

		st.write("Au final, nous obtenons un dataframe avec ", df.shape[1], "colonnes.")
		st.write("* dont ", len(df.select_dtypes('number').columns), "variables **numériques**")
		st.write("* et ", len(df.select_dtypes('object').columns), "variables **catégorielles**")

		
		########## 1ère partie Dataviz


		st.markdown("### :arrow_forward: Analyse de la variable cible : la Note du joueur au prochain match")

		# 1/ Distribution de la note à l'issue d'une journée

		st.markdown("##### 1/ Distribution de la note à l'issue d'une journée")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Ce graphique nous a permis de vérifier le côté déséquilibré de notre jeu de données (notes à 0). Puis cela nous a aidé à mieux visualiser la distribution des notes afin de créer les classes de la variable le cas échéant.")

		notes_liste = sorted(df_00['Note_Jour_Next'].unique().tolist())
		start_note, end_note = st.select_slider("> Sélectionnez les notes puis activez la lecture pour voir l'évolution des notes au cours des journées", options = notes_liste, value=[0.0,9.0])
		df1 = df_00[(df_00['Note_Jour_Next'] >= start_note) & (df_00['Note_Jour_Next'] <= end_note)]

		fig1 = px.histogram(df1, x = 'Note_Jour_Next',
							nbins = 25,
							labels = {'Note_Jour_Next':'Note de la journée'},
							animation_frame = "Journée",
							color_discrete_sequence=['#00B727'],
							width = 800
							)
		fig1.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
		fig1.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 50
		fig1.update_layout(bargap = 0.1)
		st.write(fig1)

		st.markdown("Distribution globale de la note à l'issue d'une journée, sans les notes à 0")

		fig2 = px.histogram(df_00_jour, x = 'Note_Jour_Next',
							nbins = 25,
							labels = {'Note_Jour_Next':'Note de la journée'},
							color_discrete_sequence=['#00B727'],
							width = 600
							)
		fig2.update_layout(bargap = 0.1)
		st.write(fig2)

		# 2/ Répartition des notes par match selon le résultat du match

		st.markdown("##### 2/ Répartition des notes par match selon le résultat du match")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Ce graphique nous a permis de vérifier le côté déséquilibré de notre jeu de données (notes à 0). Puis cela nous a aidé à mieux visualiser la distribution des notes afin de créer les classes de la variable le cas échéant.")

		st.image("media/charts/1-2-Répartition des notes par match selon le résultat du match.png")
		
		# 3/ Variation de la note selon le nombre de matchs joués

		st.markdown("##### 3/ Variation de la note selon le nombre de matchs joués")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Il y avait une corrélation entre la variable cible la feature 'Temps de jeu' dans la matrice. Nous avons pu vérifier que le nombre de matchs joués est corrélé positivement à la note obtenue.")

		st.image("media/charts/1-3-Variation de la note selon le nombre de matchs joués.png")


		########## 2e partie Dataviz

		st.markdown("### :arrow_forward: Analyse de la note selon le poste du joueur")

		# 1/ Distribution de la note selon le poste

		st.markdown("##### 1/ Distribution de la note selon le poste")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Dans le but de créer sa composition avec des joueurs à tous les postes, il nous semblait important de visualiser la distribution de la note selon le type de poste des joueurs.")

		st.image("media/charts/2-1-Distribution des notes selon le poste.png")

		# 2/ Evolution de la note en fonction du nombre de buts (sauf Gardiens)

		st.markdown("##### 2/ Evolution de la note en fonction du nombre de buts (sauf Gardiens)")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("L'objectif était de voir à quel point le nombre de buts marqués influent la note obtenue, et ce même pour des postes de type défensifs, intéressant !")

		df_00_gardiens = df_00[df['Poste']!='G'] # DF des joueurs avec une note, ans les Gardiens
		postes_liste = sorted(df_00_gardiens['Poste'].unique().tolist())
		postes_options = st.multiselect('Sélectionnez le(s) poste(s) pour afficher les notes des joueurs', options = postes_liste, default = ['A'])

		df2 = df_00_gardiens[df_00_gardiens['Poste'].isin(postes_options)]

		fig22 = px.scatter(df2,
							x = 'But',
							y = 'Note',
							color = 'Poste',
							symbol = 'Poste',
							trendline = "ols",
							labels = {'But':'Nombre de buts marqués', 'Note':"Note à l'issue du match"}
							)
		st.write(fig22)

		# 3/ Distribution de la note selon le poste, par club

		st.markdown("##### 3/ Distribution de la note selon le poste, par club")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Nous souhaitions avoir plus de granularité concernant la distribution de la note, en combinant le poste au club, et identifier d'éventuelles surprises.")

		st.image("media/charts/2-3-Note par club selon poste.jpg")

		# 4/ Rôle du joueur dans le match, selon le poste

		st.markdown("##### 4/ Rôle du joueur dans le match, selon le poste")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Il nous semblait important de voir quels postes étaient susceptibles d'avoir le plus de remplaçants afin de préparer sa composition en connaissances de cause.")

		st.image("media/charts/2-4-Rôle du joueur dans le match pas type de poste.png")

		# 5/ Répartition des notes par club, selon le rôle du joueur

		st.markdown("##### 5/ Répartition des notes par club, selon le rôle du joueur")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Toujours selon le rôle du joueur, l'idée est de voir si cela a un impact sur la note et si cet impact est équivalent parmi tous les clubs de L1.")

		st.image("media/charts/2-5-Répartition des notes par club, selon le rôle du joueur.png")


		########## 3e partie Dataviz

		st.markdown("### :arrow_forward: Impact du type de déplacement de l'équipe")

		# 1/ Résultats des matchs selon le déplacement des équipes

		st.markdown("##### 1/ Résultats des matchs selon le déplacement des équipes")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Graphique simple pour valider l'hypothèse selon laquelle une équipe gagne plus souvent à domicile qu'à l'extérieur.")

		st.image("media/charts/3-1-Résultats des matchs selon le déplacement des équipes.png")

		# 2/ Nombre de buts marqués par Club selon le type de déplacement

		st.markdown("##### 2/ Nombre de buts marqués par Club selon le type de déplacement")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Ce graphique montre l'importance du déplacement de l'équipe dans le nombre de buts marqués, ce qui aura une influence dans les prédictions futures selon le déplacement à venir la prochaine journée.")

		st.image("media/charts/3-2-Nombre de buts marqués par Club selon le type de déplacement.png")

		# 3/ Note des joueurs selon le poste et le déplacement de l'équipe

		st.markdown("##### 3/ Note des joueurs selon le poste et le déplacement de l'équipe")

		with st.expander("Pourquoi ce graphique ?"):
			st.write("Nous souhaitions voir si le déplacement était corrélé à la note obtenue par type de poste. Ce qui ne semble l'être, mais de façon très légère et pas de façon homogène selon le poste.")

		st.image("media/charts/3-3-Note des joueurs selon le poste et le déplacement de l'équipe.png")



################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 3/ ENTRAINEMENTS DES MODELES DE ML

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################



elif choice == "Machine learning":
	with model_training:
		st.header('3/ Machine Learning')
		
		show = st.checkbox("<< Idem, si vous souhaitez afficher le dataset original, cochez la case")
		if show:
			st.markdown("###### Voici l'extrait du dataframe original, contenant toutes les colonnes initiales.")
			st.write(df.head())

		st.write("Cette section se divise en 4 grandes parties :")
		st.write("* Choix des variables")
		st.write("* Entraînements des modèles")
		st.write("* Features engineering sur le modèle retenu")
		st.write("* ACP et conclusions")

		##############################
		# a/ Choix des variables
		##############################

		st.markdown("#### a/ Choix des variables")

		st.write(":black_medium_square: La première étape a été de réduire le nombre de variables pour le machine learning. Avec 96 variables, nous avons choisi de créer 4 matrices de corrélation pour plus de simplicité de lecture et d'analyse.")
		st.write("Voici les matrices (à agrandir pour voir les détails) de corrélation avec la variable cible **'Note_Jour_Next'** :")

		col1, col2 = st.columns(2)
		with col1:
		    st.image("media/charts/Matrice de corrélation_01.png")
		with col2:
		    st.image("media/charts/Matrice de corrélation_02.png")

		col1, col2 = st.columns(2)
		with col1:
		    st.image("media/charts/Matrice de corrélation_03.png")
		with col2:
		    st.image("media/charts/Matrice de corrélation_04.png")


		st.write(":black_medium_square: Puis, nous avons effectué plusieurs ajustements sur les variables afin de ne conserver que celles qui étaient intéressantes pour l'entraînement des modèles.")
		st.write(":point_right: Cliquez sur les boutons ci-dessous et affichez l'impact de l'action sur la forme du DF.")

		st.write("**Forme du DF initial :**", df.shape)

		df.index = df['Joueur']
		df.drop(['Joueur', 'Poste'], axis=1, inplace = True)
		if st.button("> Remplacer l'index par le nom des Joueurs"):
			st.write("Nouvelle forme du DF :", df.shape)

		to_keep = ['Côte', 'Enchère moy', 'Note', 'Variation', 'But', 'Temps', 'Jour', 'Note_Jour', 'Journée_Next',
           'Note_Jour_Next', 'Buteur', 'Entrée_Remplaçant', 'But_Match', 'Match_Noté', 'Titulaire_Match', 'Statut',
           'Derniers Résultats_avantJ', 'Status_Match_Code', 'Derniers_Status_Match_avant_J', 'Derniers_Match_Buteur_avant_J',
           'Derniers_But_Match_avant_J', 'Derniers_Status_Match_après_J', 'Derniers_Match_Buteur_après_J',
           'Derniers_But_Match_après_J', '%Buteur_Derniers_Match', '%Titulaire_Derniers_Match', '%Noté_Derniers_Match',
           '%Buteur_Derniers_Match_après', '%Titulaire_Derniers_Match_après', '%Noté_Derniers_Match_après',
           'But/Match_Derniers_Match', 'But/Match_Derniers_Match_après', 'Note_Moy_Derniers_Match',
           'Note_Moy_Derniers_Match_Dom', 'Note_Moy_Derniers_Match_Ext', 'Note_Moy_', 'Note_Moy_Derniers_Match_A',
           'Note_Moy_Derniers_Match_Dom_A', 'Note_Moy_Derniers_Match_Ext_A', 'Moy_But_Av_J', 'Nb_But_Av_J', 'Moy_But_Ap_J',
           'Nb_But_Ap_J', 'Rolling_Note_J', 'Rolling_Note_J_Ap', 'Rolling_Note_J_Derniers_Match',
           'Rolling_Note_J_Derniers_Match_Ap']
		df = df[to_keep]

		if st.button("> Exclure les variables les moins corrélées et les moins intéressantes d'un point de vue métier"):
			st.write("Nouvelle forme du DF :", df.shape)

		df = df[df['Note'] != 0]
		if st.button("> Retirer les lignes des joueurs avec une note moyenne à 0 (ie les joueurs qui ne jouent jamais)"):
			st.write("Nouvelle forme du DF :", df.shape)

		df = df[df['Jour']>5]
		if st.button("> Exclure les 5 premières journées de la saison car pas assez de data pour appliquer le modèle"):
			st.write("Nouvelle forme du DF :", df.shape)

		df = df[df['Jour'] != df['Jour'].max()]
		if st.button("> Exclure les lignes de la dernière journée en date car elles n'ont pas de valeur en variable cible"):
			st.write("Nouvelle forme du DF :", df.shape)

		df['Ratio_G_derniers matchs'] = df['Derniers Résultats_avantJ'].apply(lambda x: x.count('G')/len(x))
		df['Ratio_N_derniers matchs'] = df['Derniers Résultats_avantJ'].apply(lambda x: x.count('N')/len(x))
		df['Ratio_P_derniers matchs'] = df['Derniers Résultats_avantJ'].apply(lambda x: x.count('P')/len(x))
		df = df.drop('Derniers Résultats_avantJ',axis=1)
		if st.button("> Numériser la variable indiquant la série des derniers résultats (G, N, P)"):
			st.write("Nouvelle forme du DF :", df.shape)

		df['Statut'] = df['Statut'].replace({'Extérieur':0, 'Domicile':1})
		if st.button("> Numériser la variable indiquant le type de déplacement de l'équipe"):
			st.write("Nouvelle forme du DF :", df.shape)

		st.write("**Forme du DF final :**", df.shape)
		
		st.write(":black_medium_square: Enfin, nous avons créé les différentes classes de notre variable cible en la discrétisant.")
		st.markdown("<p style='color: black;'>Vous pouvez constater la répartition des différentes classes grâce aux curseurs ci-dessous.</br>"
											"Dans l'ordre, faites varier les <u>limites maximum</u> de chaque classe pour voir évoluer la répartition sur le graphique.</p>", unsafe_allow_html=True)

		col1, col2 = st.columns(2)
		with col1:
			note_classe0 = sorted(df['Note_Jour_Next'].unique().tolist())
			start_note0, end_note0 = st.select_slider("Limites de la classe 0", options = note_classe0, value=[0.0,1])

			note_classe1 = sorted(df['Note_Jour_Next'].unique().tolist())
			start_note1, end_note1 = st.select_slider("Limites de la classe 1", options = note_classe0, value=[end_note0,4.5])

			note_classe2 = sorted(df['Note_Jour_Next'].unique().tolist())
			start_note2, end_note2 = st.select_slider("Limites de la classe 2", options = note_classe0, value=[end_note1,6.0])

			note_classe3 = sorted(df['Note_Jour_Next'].unique().tolist())
			start_note3, end_note3 = st.select_slider("Limites de la classe 3", options = note_classe0, value=[end_note2,9.0])

		with col2:
		    df['Note_Jour_Next_Classes'] = pd.cut(x = df['Note_Jour_Next'],
                                 bins = [-1,end_note0,end_note1,end_note2,10],
                                 labels = [0, 1, 2, 3])

		    st.markdown("<p style='text-align: center; color: black;'>Répartition des classes</p>", unsafe_allow_html=True)

		    fig4 = px.pie(df,
		    	values = df['Note_Jour_Next_Classes'].value_counts(),
		    	names = ['0-Pas joué', '1-Faible', '2-Moyenne', '3-Bonne'],
		    	color_discrete_sequence = ['#8A8A8A', '#E7916B', '#FFE276', '#B8FFC1'],
		    	hole = 0.3,)
		    fig4.update_traces(textinfo='percent', textfont_size = 16)
		    st.write(fig4)

		st.write(":white_check_mark: Les bins des classes finalement retenues sont **[0, 0.5, 4.5, 6, 9]**.")
		st.write("Cela nous donne un jeu quelque peu déséquilibré, ce qui nous a conduit à mener les tests expliqués dans les parties suivantes.") 


		##############################
		# b/ Entraînements des modèles
		##############################


		st.markdown("#### b/ Entraînements des modèles")

		# Preprocessing avant entraînements

		data = df_ml.drop(['Note_Jour_Next_Classes'], axis=1)
		target = df_ml['Note_Jour_Next_Classes']
		X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, shuffle = False)
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		# Fin preprocessing


		show = st.checkbox("<< Cochez la case pour afficher l'extrait du DF final avant entraînement des modèles.")
		if show:
			st.markdown("###### Voici l'extrait du dataframe final.")
			st.write(df_ml.head())

		st.markdown("<p style='color: black;'>Dans cette partie nous présentons les tests effectués à l'aide de divers modèles de classification :</br>"
					"<ol><li>Régression logistique</li><li>K plus proches voisins</li><li>Random Forest</li><li>Decision Tree</li><li>SVC</li></ol>", unsafe_allow_html=True)


		modeles_liste = ["Choix du modèle", "Régression logistique", "K plus proches voisins", "Random Forest", "Decision Tree", "SVC"]
		modeles_choix = st.selectbox('Sélectionnez le modèle souhaité pour afficher les résultats "bruts" :', options = modeles_liste)

		if modeles_choix == "Régression logistique":
			if st.button("Lancer le modèle", key="1"):
				# my_bar = st.progress(0)
				# for i in range(100):
				# 	time.sleep(0.1)
				# 	my_bar.progress(i+1)


				lr = LogisticRegression()
				lr.fit(X_train_scaled, y_train)
				y_pred_lr = lr.predict(X_test_scaled)
				report_lr = classification_report_imbalanced(y_test, y_pred_lr, output_dict=True)
				df_lr = pd.DataFrame(report_lr).transpose()
				
				st.markdown("##### Évaluation")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.write("Score = ", lr.score(X_test_scaled, y_test).round(3))
				with col2:
					st.write("Recall = ", recall_score(y_test, y_pred_lr, average='weighted').round(3))
				with col3:
					st.write("Precision = ", precision_score(y_test, y_pred_lr, average='weighted').round(3))
				with col4:
					st.write("F1-Score = ", f1_score(y_test, y_pred_lr, average='weighted').round(3))

				st.markdown("##### Matrice de confusion")
				st.write(pd.crosstab(y_test, y_pred_lr, rownames=['Classe réelle'], colnames=['Classe prédite'])) # Matrice classique, petit tableau

				st.markdown("##### Focus sur la classe 3")
				st.write(df_lr.loc[[3], ['pre', 'rec', 'f1']].round(2))
				
		if modeles_choix == "K plus proches voisins":
			if st.button("Lancer le modèle", key="2"):
				knn = KNeighborsClassifier()
				knn.fit(X_train_scaled, y_train)
				y_pred_knn = knn.predict(X_test_scaled)
				report_knn = classification_report_imbalanced(y_test, y_pred_knn, output_dict=True)
				df_knn = pd.DataFrame(report_knn).transpose()
				
				st.markdown("##### Évaluation")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.write("Score = ", knn.score(X_test_scaled, y_test).round(3))
				with col2:
					st.write("Recall = ", recall_score(y_test, y_pred_knn, average='weighted').round(3))
				with col3:
					st.write("Precision = ", precision_score(y_test, y_pred_knn, average='weighted').round(3))
				with col4:
					st.write("F1-Score = ", f1_score(y_test, y_pred_knn, average='weighted').round(3))

				st.markdown("##### Matrice de confusion")
				st.write(pd.crosstab(y_test, y_pred_knn, rownames=['Classe réelle'], colnames=['Classe prédite'])) # Matrice classique, petit tableau

				st.markdown("##### Focus sur la classe 3")
				st.write(df_knn.loc[[3], ['pre', 'rec', 'f1']].round(2))


		if modeles_choix == "Random Forest":
			if st.button("Lancer le modèle", key="3"):
				rf = RandomForestClassifier()
				rf.fit(X_train_scaled, y_train)
				y_pred_rf = rf.predict(X_test_scaled)
				report_rf = classification_report_imbalanced(y_test, y_pred_rf, output_dict=True)
				df_rf = pd.DataFrame(report_rf).transpose()
				
				st.markdown("##### Évaluation")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.write("Score = ", rf.score(X_test_scaled, y_test).round(3))
				with col2:
					st.write("Recall = ", recall_score(y_test, y_pred_rf, average='weighted').round(3))
				with col3:
					st.write("Precision = ", precision_score(y_test, y_pred_rf, average='weighted').round(3))
				with col4:
					st.write("F1-Score = ", f1_score(y_test, y_pred_rf, average='weighted').round(3))

				st.markdown("##### Matrice de confusion")
				st.write(pd.crosstab(y_test, y_pred_rf, rownames=['Classe réelle'], colnames=['Classe prédite'])) # Matrice classique, petit tableau

				st.markdown("##### Focus sur la classe 3")
				st.write(df_rf.loc[[3], ['pre', 'rec', 'f1']].round(2))


		if modeles_choix == "Decision Tree":
			if st.button("Lancer le modèle", key="4"):
				dt = DecisionTreeClassifier()
				dt.fit(X_train_scaled, y_train)
				y_pred_dt = dt.predict(X_test_scaled)
				report_dt = classification_report_imbalanced(y_test, y_pred_dt, output_dict=True)
				df_dt = pd.DataFrame(report_dt).transpose()
				
				st.markdown("##### Évaluation")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.write("Score = ", dt.score(X_test_scaled, y_test).round(3))
				with col2:
					st.write("Recall = ", recall_score(y_test, y_pred_dt, average='weighted').round(3))
				with col3:
					st.write("Precision = ", precision_score(y_test, y_pred_dt, average='weighted').round(3))
				with col4:
					st.write("F1-Score = ", f1_score(y_test, y_pred_dt, average='weighted').round(3))

				st.markdown("##### Matrice de confusion")
				st.write(pd.crosstab(y_test, y_pred_dt, rownames=['Classe réelle'], colnames=['Classe prédite'])) # Matrice classique, petit tableau

				st.markdown("##### Focus sur la classe 3")
				st.write(df_dt.loc[[3], ['pre', 'rec', 'f1']].round(2))


		if modeles_choix == "SVC":
			if st.button("Lancer le modèle", key="5"):
				svc = SVC()
				svc.fit(X_train_scaled, y_train)
				y_pred_svc = svc.predict(X_test_scaled)
				report_svc = classification_report_imbalanced(y_test, y_pred_svc, output_dict=True)
				df_svc = pd.DataFrame(report_svc).transpose()
				
				st.markdown("##### Évaluation")
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.write("Score = ", svc.score(X_test_scaled, y_test).round(3))
				with col2:
					st.write("Recall = ", recall_score(y_test, y_pred_svc, average='weighted').round(3))
				with col3:
					st.write("Precision = ", precision_score(y_test, y_pred_svc, average='weighted').round(3))
				with col4:
					st.write("F1-Score = ", f1_score(y_test, y_pred_svc, average='weighted').round(3))

				st.markdown("##### Matrice de confusion")
				st.write(pd.crosstab(y_test, y_pred_svc, rownames=['Classe réelle'], colnames=['Classe prédite'])) # Matrice classique, petit tableau

				st.markdown("##### Focus sur la classe 3")
				st.write(df_svc.loc[[3], ['pre', 'rec', 'f1']].round(2))



		##############################
		# c/ Features engineering sur le modèle retenu
		##############################

		st.markdown("#### c/ Features engineering et optimisations sur le modèle retenu")
		st.write("Nous avons choisi de retenir le modèle **SVC** pour la suite des tests, étant celui qui obtenait les meilleurs scores bruts.") 
		st.write("Voici les différentes étapes suivies pour tenter d'optimiser le modèle retenu :") 

		st.image("media/tableau_features.PNG")

		st.markdown("<p style='color: black;'><b>Constats :</b></br>"
					"<ol><li>Le modèle 'brut' conserve le meilleur score d'accuracy</li>"
					"<li>Tous les tests ont permsi d'améliorer nettement le F1 de la classe 3</li>"
					"<li>L'ajout des 3 hyperparamètres permet de conserver un score correct et d'améliorer fortement le F1</li>"
					"<li>Les tests de feature engineering ont donné de bons résultats, mais qui restaient moins significatifs </li>"
					"<li>SVC</li>"
					"</ol>", unsafe_allow_html=True)

		st.write("**CONCLUSION**") 
		st.write("Les meilleures performances sont obtenues grâce à l'ajout des 3 hyperparamètres, sans besoin de modifier le jeu de données initial.") 


		##############################
		# d/ ACP
		##############################


		st.markdown("#### d/ ACP")
		st.write("Après cette phase de tests, nous avons tout de même souhaité mené une Analyse de Composantes Principales afin de voir si nous pouvions réduire le scope de l'analyse.") 

		st.write("Les différentes analyses menées (ACP seule, T-SNE seule, puis la combinaison ACP + T-SNE) nous indiquent que nous ne pouvons pas identifier 2 ou 3 composantes qui pourraient expliquer la variance de façon suffisante.") 
		st.write("Parmi nos 49 features, il en faudrait **au moins 15** pour expliquer 95% !")

		st.write("**Pour visualiser les graphiques explicatifs de l'analyse ACP seule, cliquez sur les blocs ci-dessous :**")

		with st.expander("Afficher le graphique de la variance expliquée"):
			st.image("media/charts/Affichage de la variance expliquée.png")

		with st.expander("Afficher le graphique du ratio de variance expliquée"):
			st.image("media/charts/Affichage du ratio de variance expliquée.png")

		with st.expander("Afficher le graphique de parts de variance expliquée"):
			st.image("media/charts/Part de variance expliquée par chaque PC.png")

		st.write("L'analyse nous donne aussi une indication quant à la répartition des classes selon les deux premières dimensions. On voit sur les graphiques que les 3 classes 1 / 2 / 3 sont assez mélangées entre elles. Mais on distingue une séparation entre ces 3 classes et la classe 0, majoritaire.")

		with st.expander("Afficher les 3 graphiques de parts de répartition des classes : ACP / T-SNE / ACP+T-SNE"):
			st.image("media/charts/Répartition des classes de la variable cible selon les deux premières dimensions - ACP.png")
			st.image("media/charts/Répartition des classes de la variable cible selon les deux premières dimensions - T-SNE.png")
			st.image("media/charts/Répartition des classes de la variable cible selon les deux premières dimensions - ACP+T-SNE.png")

		st.write("Le jeu de données initial étant quelque peu déséquilibré, l'ACP en est également compliquée. De plus, une particularité métier rend l'analyse encore plus complexe : il y a autant de lignes pour un seul joueur que de journée, ce qui veut dire que pour un même joueur, on peut avoir beaucoup de variantes dans les classes. Un joueur peut osciller entre les classes 0-1-2-3 sur 4 journées différentes par exemple.")
		st.write("Si l'exercice était intéressant d'un point de vue pédagogique, il ne nous apporte pas d'éléments significatifs pour la poursuite de la modélisation.")

		##############################
		# e/ CONCLUSIONS
		##############################

		st.markdown("#### e/ CONCLUSIONS")
		st.write("Nous avons pu mener de nombreux tests sur plusieurs items et obtenons, in fine, un modèle qui va nous permettre de proposer de nouveaux outils aux joueurs de MPG. A découvrir dans la section suivante !")


################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

# 4/ TESTS REELS

################################################################################################################################################################################################################################
################################################################################################################################################################################################################################

elif choice == "Outils MPG":
	with tools:
            st.header('4/ Outils MPG')
            st.write("MPG c'est une expérience ludique très sympathique mais pour les non-footeux et/ou ceux qui jouent pour gagner, \
                     la perspective de ne pas savoir comment exploiter le plein potentiel de son équipe peu devenir un peu frustant. \n \
                    \n Heureusement, on a penser à quelques outils qui pourraient bien te faire expérimenter le meilleur d'une compétition MPG.\
                        \n ")
            ##############################
            st.markdown("###  Auto-Mercato")
        
        
            st.write("Besoin d'aide pour monter une équipe? On va te générer une équipe MPG automatiquement")
            col1, col2, col3,col4,col5 = st.columns(5)
            with col2:
                st.image("https://media.giphy.com/media/nP1EHm9A7M5QONQSKr/giphy.gif?cid=ecf05e473wxsnkgaypwmy4xp46qq9ve9xev0zfimcnqopla2", # I prefer to load the GIFs using GIPHY
                            width=200, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
                        )
            
            df_MPG_Current=df0
            data=df_MPG_Current[['Joueur','Poste_bis','Poste','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')#on conserve 1 ligne par joueur
            
            data_pick=data#
            MPG_budget = st.slider("Quel est votre budget pour ce mercato?",300,700,500)
            budget = MPG_budget
            liste=[]#initialiser une liste vide
            nb_pick=0 # initialiser un counter 
            #diviser le panel de joueurs par poste
            data_pick_A=data_pick[data_pick['Poste_bis']=='A']
            data_pick_D=data_pick[data_pick['Poste_bis']=='D']
            data_pick_M=data_pick[data_pick['Poste_bis']=='M']
            data_pick_G=data_pick[data_pick['Poste_bis']=='G']
            
            #définir des valeurs par défaut pour éviter les messages d'erreurs en cas de non affichage de certains éléments
            repartition_budget="_"
            options_budgets=['Egale','Selon le nb de postes',"Priorité à l'attaque", "Priorité au milieu","Priorité à la défense"]
            budget_A,budget_D,budget_M,budget_G= int(budget*0.5),int(budget*0.2),int(budget*0.2),int(budget*0.1)
            nb_A,nb_D,nb_M,nb_G=4,6,6,2

            my_selection=[]
            if st.checkbox('Vérifier les paramètres pour générer votre équipe?'):
                #on définit le nb de joueurs souhaités
                nb_A=st.slider("Combien d'attaquants pour votre équipe (idéalement)?",4,10,4)
                nb_D=st.slider("Combien de défenseurs pour votre équipe (idéalement)?",6,12,6)
                nb_M=st.slider("Combien de milieux pour votre équipe (idéalement)?",6,12,6)
                nb_G=st.slider("Combien de gardiens pour votre équipe (idéalement)?",2,4,2)
                nb_Postes=nb_A+nb_M+nb_D+nb_G
                #tactique = st.selectbox(
                 #'Schéma tactique?',
                 #('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
                
                #on définit la strégie de mercato
                #options_budgets=['Egale','Selon le nb de postes',"Priorité à l'attaque", "Priorité au milieu","Priorité à la défense"]
                #La stratégie de marcato va attribuer une fraction du budget total pour chaucn des postes à pourvoir
                repartition_budget = st.selectbox('Répartition du budget (A-M-D-G)?',options_budgets)
                
            if repartition_budget==options_budgets[0]:
                budget_A,budget_M,budget_D,budget_G= int(budget*0.25),int(budget*0.25),int(budget*0.25),int(budget*0.25)
            elif repartition_budget==options_budgets[1]:
                budget_A,budget_M,budget_D,budget_G= int(budget*(nb_A/nb_Postes)),int(budget*(nb_M/nb_Postes)),int(budget*(nb_D/nb_Postes)),int(budget*(nb_G/nb_Postes))
            elif repartition_budget==options_budgets[2]:
                budget_A,budget_M,budget_D,budget_G= int(budget*0.5),int(budget*0.2),int(budget*0.2),int(budget*0.1)
            elif repartition_budget==options_budgets[3]:
                budget_A,budget_M,budget_D,budget_G= int(budget*0.25),int(budget*0.5),int(budget*0.2),int(budget*0.05)
            elif repartition_budget==options_budgets[4]:
                budget_A,budget_M,budget_D,budget_G= int(budget*0.25),int(budget*0.2),int(budget*0.5),int(budget*0.05)
            else :
                budget_A,budget_D,budget_M,budget_G= int(budget*0.5),int(budget*0.2),int(budget*0.2),int(budget*0.1)
            
            
            # Ligne par ligne , on sélectionne des joueurs et on réduit le budget, l'excédent de budget éventuel est conservé pour les autres postes
            # on sélectionne au hasard , avec une probabilité accrue selon le niveau d'enchère moy, afin de générer des équipes MPG vraisemblables
            #les joueurs sont valorisés à leur valeur d'enchère ( et non à leur côte), des règles différentes de valorisation auraient pu être retenues
            
            #Le principe général de l'algorithme d'auto-mercato
            #Tant qu'il reste du budget dédié et que le nb de de joueurs minimum pour le poste ,n'est pas atteint, on sélectionne aléatoirement un joueur
            #La probabilité est pondérée selon la valeur de l'enchère moyenne, pour simuler la désirabilité dé chaque joueur
            #Chaque joueur ainsi sélectionné est rajouté retirer du pool de joueur et le nb de joueurs à sékectionner et décrémenter
            #si le joueur sélectionné est trop cher, tout les joeurs de sa valeurs sont retiré du pool
            # si il reste encore des joeurs à sélectioner mai splsu de budget disponible alors une rallonge de budget est accordée
            # a l'inverse si il rest du budget,il est conserver pour les autres postes
            #l'opération ce repète poru l'ensemble des postes
            #le reliquat de buget éventuel est consommé un sélectionnant des joueurs non encore sélectionnés jusqu'à épuisemment des fonds disponibles
            #Cet algorithme imite le process de réflexion possible sur MPG et abouti à des équipe vraisemblable
            
            
            if st.button("Générer équipe?"):
            
                liste_A=[]
                adj=0
                nb_pick=0 
                #budget_A=int(budget*0.5)
                while (budget_A > 0 and nb_pick < nb_A+4):
                    pick=data_pick_A.sample(n=1,weights="Enchère moy")
                    if budget_A >= pick["Enchère moy"].sum():
                        liste_A +=pick["Joueur"].tolist()
                        budget_A -=pick["Enchère moy"].sum()
                        data_pick_A=data_pick_A.drop(pick.index)
                        nb_pick +=1
                        if nb_pick<4 and budget_A <1 :
                            budget_A=1
                            adj-=1
                    else :
                        data_pick_A=data_pick_A[data_pick_A["Enchère moy"]<pick["Enchère moy"].sum()]
                
                budget_M=budget_M+budget_A         
                liste_M=[]
                nb_pick=0 
                while (budget_M > 0 and nb_pick < nb_M+6):
                    pick=data_pick_M.sample(n=1,weights="Enchère moy")
                    if budget_M >= pick["Enchère moy"].sum():
                        liste_M +=pick["Joueur"].tolist()
                        budget_M -=pick["Enchère moy"].sum()
                        data_pick_M=data_pick_M.drop(pick.index)
                        nb_pick +=1
                        if nb_pick<6 and budget_M <1 :
                            budget_M+=1
                            adj-=1
                    else :
                        data_pick_M=data_pick_M[data_pick_M["Enchère moy"]<pick["Enchère moy"].sum()] 
                
                nb_pick=0        
                budget_D=budget_D+budget_M       
                liste_D=[]
                while (budget_D > 0 and nb_pick < nb_D+6):
                    pick=data_pick_D.sample(n=1,weights="Enchère moy")
                    if budget_D >= pick["Enchère moy"].sum():
                        liste_D +=pick["Joueur"].tolist()
                        budget_D -=pick["Enchère moy"].sum()
                        data_pick_D=data_pick_D.drop(pick.index)
                        nb_pick +=1
                        if nb_pick<6 and budget_D < 1 :
                            budget_D=1
                            adj-=1
                    else :
                        data_pick_D=data_pick_D[data_pick_D["Enchère moy"]<pick["Enchère moy"].sum()]        
                
                
                budget_G=budget_G+budget_D+adj
                liste_G=[]
                nb_pick=0 
                while (budget_G > 0 and nb_pick < nb_G+2):
                    pick=data_pick_G.sample(n=1,weights="Enchère moy")
                    if budget_G >= pick["Enchère moy"].sum():
                        liste_G +=pick["Joueur"].tolist()
                        budget_G -=pick["Enchère moy"].sum()
                        data_pick_G=data_pick_G.drop(pick.index)
                        nb_pick +=1
                    else :
                        data_pick_G=data_pick_G[data_pick_G["Enchère moy"]<pick["Enchère moy"].sum()]
                        
                listes= liste_A+liste_D+liste_M+liste_G        
                budget_restant=budget_G
                st.write(budget_restant)
                if budget_restant>10 :
                    liste_restant=[]
                    nb_pick=0 
                    data_pick_restant=data[~data['Joueur'].isin(listes)]
                    
                    while (budget_restant > 0):
                        pick=data_pick_restant.sample(n=1,weights="Enchère moy")
                        if budget_restant >= pick["Enchère moy"].sum():
                            liste_restant +=pick["Joueur"].tolist()
                            budget_restant -=pick["Enchère moy"].sum()
                            data_pick_restant=data_pick_restant.drop(pick.index)
                            nb_pick +=1
                            #st.write(budget_restant)
                        else :
                            data_pick_restant=data_pick_restant[data_pick_restant["Enchère moy"]<pick["Enchère moy"].sum()]        
                    listes= listes+liste_restant
                
                #on cumule les différentes listes       
                #listes= liste_A+liste_D+liste_M+liste_G
                #listes= listes+liste_restant      
                mon_equipe=data[data['Joueur'].isin(listes)].sort_values("Poste_bis",ascending=False)
                
                #on vérifie l'auto-mercato
                functions_to_apply = {
                    'Poste' : ['count'],
                    'Enchère moy':['sum']
                }
                df =mon_equipe.groupby(['Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
                df.columns=['Poste','nb','sum']
                df.loc[4]= df.sum(numeric_only=True,axis=0)
                df=df.fillna("_")
                df.loc[4,'Poste']='Total'
                df['nb']=df['nb'].map('{:,.0f}'.format)
                df['sum']=df['sum'].map('{:,.0f}'.format)
                
                total=len(mon_equipe['Joueur'].to_list())
                avg_note=mon_equipe['Note'].mean()
                mon_equipe=mon_equipe.reset_index(drop=True)
                mon_equipe.loc[total]= mon_equipe.sum(numeric_only=True, axis=0)
                mon_equipe.loc[total,'Note']=avg_note
                mon_equipe.loc[total,'Joueur']='total'
                mon_equipe=mon_equipe.fillna("_")
                mon_equipe['Côte']=mon_equipe['Côte'].map('{:,.0f}'.format)
                mon_equipe['Enchère moy']=mon_equipe['Enchère moy'].map('{:,.0f}'.format)
                mon_equipe['Note']=mon_equipe['Note'].map('{:,.2f}'.format)
                
                st.write("Alors, le hasard fait-il bien les choses ? 🥳 ou 😅")
                st.dataframe(df)
                st.dataframe(mon_equipe)
                my_selection=listes
                save_selection=pd.DataFrame(my_selection)
                save_selection.to_csv("media/saved_selection.csv",header=False,index=False)
                
                ##############################
                
                st.header("Mon Equipe")#paragraphe
                st.write("On affiche son équipe MPG")

                #selection de joueurs en dur, si pas de sélection auto-générée
                my_selection_0=['Baldé Ibrahima',
                 'Nguette Opa',
                 'Guirassy Serhou',
                 'Milik Arkadiusz',
                 'Mbuku Nathanaël',
                 'Weah Timothy',
                 'Laborde Gaëtan',
                 'Mikelbrencis William',
                 'Denayer Jason',
                 'Andrei Girotto',
                 'Lirola Pol',
                 'Abdelhamid Yunis',
                 'Jemerson',
                 'Dante',
                 'Seidu Alidu',
                 'Oyongo Ambroise',
                 'Sissokho Issouf',
                 'Payet Dimitri',
                 'Chirivella Pedro',
                 'Bamba Jonathan',
                 'Agoume Lucien',
                 'Prcic Sanjin',
                 "M'Bock Hianga'a",
                 'Thomasson Adrien',
                 'Doucet Lohann',
                 'Bitumazala Nathan',
                 'Donnarumma Gianluigi',
                 'Petkovic Danijel',
                 'Navas Keylor']
                
                
                
                if my_selection==[]:
                    #my_selection=my_selection_0
                    my_selection=pd.read_csv('media/saved_selection.csv',header=None)[0].values.tolist()
                    #st.write("Vous n'avez pas généré d'équipe ou avez modifié certains paramètres de l'auto-mercato sans regénérer d'équipe, d'où l'affichage d'une équipe par défaut ")
                else :
                    st.write("Votre équipe auto-générée")
                
                
                
                data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
                data=data.drop_duplicates(keep = 'first')
                data=data[data['Joueur'].isin(my_selection)]
                data=data.sort_values(by=['Note'],ascending=False)
                data.index = np.arange(1, len(data) + 1)
                #st.dataframe(data.round(2))
                #Pour afficher ou non, son équipe, selon son schéma tactique
                
                tactique="4-4-2"
                #tactique = st.selectbox(
                #'Schéma tactique?',
                #('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
                tactique_D,tactique_M,tactique_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
                
                
                #if st.checkbox('afficher son équipe?'):
                    #st.write("On affiche son équipe MPG")
                    #tactique = st.selectbox(
                     #'Schéma tactique?',
                     #('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
                    #tactique_D,tactique_M,tactique_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
                    
                    #row_G,row_D,row_M,row_A=st.container(),st.container(),st.container(),st.container()
                my_expander = st.expander(label='afficher mon équipe?',expanded=True)
                
                 
                with my_expander:
                    
                 
                    with st.container():#ligne gardien
                        col1, col2, col3,col4,col5 = st.columns(5)
                        local_G=data[data['Poste_bis']=='G']["Joueur"]
                        list_club=data[data['Poste_bis']=='G']["Club"]
                        with col3:
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_G.iloc[0]
                            st.image(img,caption=label,width=80)
                            
                    with st.container(): #défense
                        local_D=data[data['Poste_bis']=='D']["Joueur"]
                        list_club=data[data['Poste_bis']=='D']["Club"]
                        #l=glob.glob("Maillots/"+list_club+".png")
                        if tactique_D==3 :
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col2 :
                                #st.write(local_D.iloc[0])
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)    
                        elif tactique_D==4:
                            col1, col2, col3,col4 = st.columns(4)
                            with col1 :
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col2 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                                label=local_D.iloc[3]
                                st.image(img,caption=label,width=80)
                        else:
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col1 :
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col2 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                                label=local_D.iloc[3]
                                st.image(img,caption=label,width=80) 
                            with col5 :
                                img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                                label=local_D.iloc[4]
                                st.image(img,caption=label,width=80)
                
                    with st.container():#milieu
                        local_D=data[data['Poste_bis']=='M']["Joueur"]
                        list_club=data[data['Poste_bis']=='M']["Club"]
                        #l=glob.glob("Maillots/"+list_club+".png")
                        if tactique_M==3 :
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col2 :
                                #st.write(local_D.iloc[0])
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)    
                        elif tactique_M==4:
                            col1, col2, col3,col4 = st.columns(4)
                            with col1 :
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col2 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                                label=local_D.iloc[3]
                                st.image(img,caption=label,width=80)
                        else:
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col1 :
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col2 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                                label=local_D.iloc[3]
                                st.image(img,caption=label,width=80) 
                            with col5 :
                                img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                                label=local_D.iloc[4]
                                st.image(img,caption=label,width=80) 
                                
                    with st.container():#attaque
                        local_D=data[data['Poste_bis']=='A']["Joueur"]
                        list_club=data[data['Poste_bis']=='A']["Club"]
                        #l=glob.glob("Maillots/"+list_club+".png")
                        if tactique_A==3 :
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col2 :
                                #st.write(local_D.iloc[0])
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col3 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                                label=local_D.iloc[2]
                                st.image(img,caption=label,width=80)    
                        elif tactique_A==1 :
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col3 :
                                #st.write(local_D.iloc[0])
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                        else:
                            col1, col2, col3,col4,col5 = st.columns(5)
                            with col2 :
                                img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                                label=local_D.iloc[0]
                                st.image(img,caption=label,width=80)
                            with col4 :
                                img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                                label=local_D.iloc[1]
                                st.image(img,caption=label,width=80)
                    
                    
            ##############################        
                    
            st.markdown("###  Afficher son équipe")
        
        
            #st.header("Charger son équipe")#paragraphe
            st.write("Tu as une équipe MPG, fait voir à quoi elle ressemble!")          
            uploaded_file=None #uploaded_file = st.file_uploader("Sélectionne un fichier sur ton ordinateur")
            if uploaded_file is  None:
                my_selection=pd.read_csv('media/saved_selection.csv',header=None)[0].values.tolist()
                st.write("On part sur la dernière équipe auto-générée, sauf si tu veux en générer une autre?")
            
            if uploaded_file is not None:
                my_selection=pd.read_csv(uploaded_file,header=None)[0].values.tolist()
                st.write("Ok, on part sur cette liste de joueurs.")
                
            data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')
            data=data[data['Joueur'].isin(my_selection)]
            data=data.sort_values(by=['Note'],ascending=False)
            data.index = np.arange(1, len(data) + 1)
            #st.dataframe(data.round(2))
            #Pour afficher ou non, son équipe, selon son schéma tactique
            
            tactique = st.selectbox(
            'Schéma tactique?(2)',
            ('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
            tactique_D,tactique_M,tactique_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
            
            
            #if st.checkbox('afficher son équipe?'):
                #st.write("On affiche son équipe MPG")
                #tactique = st.selectbox(
                 #'Schéma tactique?',
                 #('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
                #tactique_D,tactique_M,tactique_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
                
                #row_G,row_D,row_M,row_A=st.container(),st.container(),st.container(),st.container()
            my_expander = st.expander(label='afficher mon équipe?',expanded=True)
            
             
            with my_expander:
                with st.container():#ligne gardien
                    col1, col2, col3,col4,col5 = st.columns(5)
                    local_G=data[data['Poste_bis']=='G']["Joueur"]
                    list_club=data[data['Poste_bis']=='G']["Club"]
                    with col3:
                        img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                        label=local_G.iloc[0]
                        st.image(img,caption=label,width=80)
                        
                with st.container(): #défense
                    local_D=data[data['Poste_bis']=='D']["Joueur"]
                    list_club=data[data['Poste_bis']=='D']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_D==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_D==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80)
                
                with st.container():#milieu
                    local_D=data[data['Poste_bis']=='M']["Joueur"]
                    list_club=data[data['Poste_bis']=='M']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_M==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_M==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80) 
                            
                with st.container():#attaque
                    local_D=data[data['Poste_bis']=='A']["Joueur"]
                    list_club=data[data['Poste_bis']=='A']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_A==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_A==1 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col3 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)

            ##############################

            st.markdown("###  Comparer mes joueurs")
            
            st.write("Comparons les joueurs!📊")
            st.write("Quelles sont les pépites de l'effectif")
            
            col1, col2, col3,col4,col5 = st.columns(5)
            with col2:
                st.image("https://media0.giphy.com/media/LnYiJhfVhywlrzF7jo/giphy.gif", # I prefer to load the GIFs using GIPHY
                            width=200, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
                        )
            
            st.write("Pour les besoins de la démonstration, on va utiliser notre équipe. Mais çà fonctionnerait pareil pour la tienne.😉")
            
            my_expander_compo1 = st.expander(label="Ma liste de joueur-Compo1",expanded=False)
            
            with my_expander_compo1: 
                compo_1=pd.read_csv('data/Compo_1.csv',header=None)
                compo_1.columns=["Joueur"]
                st.write("Notre liste de joueurs")
                st.dataframe(compo_1)
            
            mon_jour = st.slider("Données entre les journées ...",value=[1,38],min_value=1,max_value=38)
            check = st.checkbox('Exclure les matchs sans notes?')   
            sort_option = st.selectbox('Trier par?',('Nom', 'Note', 'Poste'))
            
            my_expander_1 = st.expander(label='Voir la comparaison?',expanded=True)
            with my_expander_1:
                #mon_jour = st.slider("Données entre les journées ...",value=[1,38],min_value=1,max_value=38)
                #check = st.checkbox('Exclure les matchs sans notes?')   
                #sort_option = st.selectbox('Trier par?',('Nom', 'Note', 'Poste'))
                
                
                functions_to_apply = {
                        'Note_Jour' : ['mean','count']
                    }
                    
                compo_1=pd.read_csv('data/Compo_1.csv',header=None)
                compo_1.columns=["Joueur"]
                df_compo_1=df0.loc[df0["Joueur"].isin(compo_1["Joueur"])]
                if check:
                    df_graph=df_compo_1[(df_compo_1['Note_Jour']!=0)&(df_compo_1['Jour']>=mon_jour[0])&(df_compo_1['Jour']<=mon_jour[1])]
                else :
                    df_graph=df_compo_1[(df_compo_1['Note_Jour']>=0)&(df_compo_1['Jour']>=mon_jour[0])&(df_compo_1['Jour']<=mon_jour[1])]
                #df_graph=df_compo_1
                data =df_graph.groupby(['Club','Joueur','Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
                data.columns=['Club','Joueur','Poste','Note','nb_Note']
                data['Joueur-Poste']=data["Joueur"]+"("+data["Poste"]+")"
                
                data=data.sort_values(by=['Joueur'])
                if sort_option=="Note":
                    data=data.sort_values(by=['Note'],ascending=False)
                elif sort_option=='Poste':    
                    data=data.sort_values(by=['Poste'])
                
                
                #data.sort_values(by=['Note'],ascending=False,inplace=True)
                
                fig=plt.figure(figsize=(30, 30))
                
                #unique = data['Club'].unique()
                list_club=data['Club'].to_list()
                
                
                palette={'A': sns.color_palette("hls",4)[0],
                 'D': sns.color_palette("hls",4)[1],
                 'M': sns.color_palette("hls",4)[2],
                 'G': sns.color_palette("hls",4)[3]}
                
                data["couleur"]='blue'
                
                data.loc[data['Poste']=='D',"couleur"]='#45f542'
                data.loc[data['Poste']=='M',"couleur"]='#4278f5'
                data.loc[data['Poste']=='G',"couleur"]='grey'
                list_color=data['couleur'].to_list()
                
                
                sns.catplot(x="Joueur-Poste", y="Note", kind="bar", height=8, aspect=2.0,alpha=0.5,palette=list_color,  data=data,legend=False)
                
                plt.title('Note moyenne de votre équipe',fontsize=14);
                
                plt.plot(range(len(list_club)),[data[data['Poste']=='A']['Note'].mean()] * len(list_club),color='blue',ls='--',alpha=0.5,label="Moy_A")
                plt.plot(range(len(list_club)),[data[data['Poste']=='D']['Note'].mean()] * len(list_club),color='#45f542',ls='--',alpha=0.5,label="Moy_D")
                plt.plot(range(len(list_club)),[data[data['Poste']=='M']['Note'].mean()] * len(list_club),color='#4278f5',ls='--',alpha=0.5,label="Moy_M")
                plt.plot(range(len(list_club)),[data[data['Poste']=='G']['Note'].mean()] * len(list_club),color='grey',ls='--',alpha=0.5,label="Moy_G")
                
                
                height=0.08
                witdh=0.2
                maxy=data['Note'].max()
                miny=data['Note'].min()+0.1
                
                for i in np.arange(len(list_club)):
                    l=glob.glob(path_maillots+list_club[i]+".png")
                    img=plt.imread(l[0])
                    plt.imshow(img,aspect='auto',extent=[i-witdh, i+witdh, miny- height, miny + height], zorder=2)
                
                maxy=data['Note'].max()
                
                plt.xlim(-1, len(list_club))
                plt.ylim(miny-0.1,maxy+0.5)    
                plt.ylabel('Note moyenne',fontsize=12)
                plt.xlabel('Joueur',fontsize=12)
                plt.tick_params(labelsize=8)
                plt.legend()
                plt.tight_layout()    
                  
                plt.xticks(rotation = 90)
                plt.show();
                st.pyplot()
                
            ##############################
            
            st.markdown("###  Comparer les statistiques")
            st.write("Qui a les meilleures stats?")
            
            postes=["G","D","M","A"]
            postes_options = st.selectbox(
                 'Sélectionner poste',
                 postes)
            
            
            
            
            
            
            data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')
            data=data[data['Joueur'].isin(my_selection)]
            
            
            functions_to_apply = {
                    'Note_Jour' : ['mean','count','sum'],
                    'Match_Noté':["sum"],
                    'Titulaire_Match':['sum'],
                    'But_Match':['sum']
                }
            df_graph=df_compo_1[(df_compo_1['Note_Jour']>=0)&(df_compo_1['Jour']<=38)]
            #df_graph=df_compo_1
            data =df_graph.groupby(['Club','Joueur','Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
            data.columns=['Club','Joueur','Poste','Note','nb_Match','sum_note','nb_Note','nb_tit',"nb_But"]
            data["But/Match noté"]=data["nb_But"]/data["nb_Note"]
            data["But/Match noté"]=data["But/Match noté"].round(2)
            data["%Titulaire_noté"]=data["nb_tit"]/data["nb_Note"]
            data["%Titulaire_noté"]=data["%Titulaire_noté"].round(2)
            data["%Titulaire"]=data["nb_tit"]/data["nb_Match"]
            data["%Titulaire"]=data["%Titulaire"].round(2)
            data["Note2"]=data["sum_note"]/data["nb_Note"]
            data["Note2"]=data["Note2"].round(2)
            data=data.fillna(0)
            data=data.drop("sum_note",axis=1)
            
            
            
            df=data[data["Poste"]==postes_options][["Joueur",'Club','%Titulaire','Note','nb_But']]
            df.columns=['Joueur','Club','%Titulaire','Note','nb_But']
            df=df.melt(id_vars=['Joueur',"Club"],value_vars=['%Titulaire', 'Note',"nb_But"],var_name='KPI', value_name='Value')
            
            
            my_expander_stats = st.expander(label="afficher les stats?",expanded=False)
            with my_expander_stats:
                st.dataframe(df.style.format('{:.2f}', subset=['Value']))
            
            my_expander_stats_graph = st.expander(label="afficher les stats (graphiques?",expanded=False)
            
            with my_expander_stats_graph:
                fig=plt.figure(figsize=(21, 21))
                #palette=['green','blue','red']
                sns.catplot(x="KPI", y="Value", kind="bar", height=8, aspect=1.0,alpha=0.8,hue="Joueur",  data=df,legend=True,col='KPI',sharex=False,sharey=False,col_wrap=3)
                #sns.catplot(x="KPI", y="Value", kind="bar", height=8, aspect=2.0,alpha=0.5,hue="Joueur",  data=df_1,legend=True)
                
                #plt.title('Comparaison sur les statistiques importantes',fontsize=14,loc='center');
                plt.show()
                st.pyplot()
            
            
            ##############################
            
            st.markdown("###  Retour sur le match précédent")
            st.write("Quelle aurait été la meilleure compo possible ?")
            
            col1, col2, col3,col4,col5 = st.columns(5)
            with col2:
                st.image("https://media4.giphy.com/media/tvk6Of5sqgXbgBv5Iq/giphy.gif", # I prefer to load the GIFs using GIPHY
                            width=200, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
                        )
            
            dernière_J = st.slider("Afficher la meilleure compo de la journée",1,38,33)
            nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
            nb_team=0
            
            my_expander_debrief = st.expander(label="afficher le debrief? Journée "+str(dernière_J),expanded=True)
            
            with my_expander_debrief: 
                functions_to_apply = {
                    'Note_Jour' : ['mean'],
                    'Match_Noté':["sum"],
                    'Titulaire_Match':['sum'],
                    'But_Match':['sum']
                }
                compo_1=pd.read_csv('data/Compo_1.csv',header=None)
                compo_1.columns=["Joueur"]
                df_compo_1=df0.loc[df0["Joueur"].isin(compo_1["Joueur"])]
                
                df_graph=df_compo_1[df_compo_1['Jour']==dernière_J]
                #df_graph=df_compo_1
                data =df_graph.groupby(['Club','Joueur','Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
                data.columns=['Club','Joueur','Poste','Note','nb_Note','nb_tit',"nb_But"]
                data["But/Match noté"]=data["nb_But"]/data["nb_Note"]
                data["But/Match noté"]=data["But/Match noté"].round(2)
                data["%Titulaire_noté"]=data["nb_tit"]/data["nb_Note"]
                data["%Titulaire_noté"]=data["%Titulaire_noté"].round(2)
                data["%Titulaire"]=data["nb_tit"]/data["nb_Note"]
                data["%Titulaire"]=data["%Titulaire"].round(2)
                data=data.fillna(0)
                data.sort_values(by=['Note'],ascending=False,inplace=True)
                #st.dataframe(data)
                
                #méthode itérative pour récupérer les positions 1 à 1 et en déduire le schéma tactique
                nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
                nb_team=0
    
                for p in data["Poste"] :
                    if nb_team <=8 :
                        if (p=="D") & (nb_D<5):
                            nb_D +=1
                            nb_team +=1
                        elif (p=="M") & (nb_M<5):
                            nb_M +=1
                            nb_team +=1
                        elif (p=="A") & (nb_A<3):
                            nb_A +=1
                            nb_team +=1 
                            
                        #print(nb_D,nb_M,nb_A)
                    elif nb_team <10:
                        if (nb_D==4) and (nb_M==2):
                            nb_M +=1
                            nb_team +=1
                            
                        elif (nb_D==2) and (nb_M==5):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==2) and (nb_M==4):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==3) and (nb_M==3):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==5) and (nb_M==2):
                            nb_M +=1
                            nb_team +=1
                        elif (nb_D==4) and (nb_M==4):
                            nb_A +=1
                            nb_team +=1
                        elif (nb_D==3) and (nb_M==4):
                            nb_D +=1
                            nb_team +=1    
                        elif (nb_D==5) and (nb_M==1):
                            nb_M =3
                            nb_D=4
                            nb_A=3
                            nb_team +=1    
                        elif (nb_D==5) and (nb_M==3):
                            nb_A +=1
                            nb_team +=1
                                      
                #on récuprère les lignes de chaques postes
                        
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.head(1)
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.head(5)
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.head(5)
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.head(3)
                
                #on ne conserve que les premières lignes nécessaires
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.iloc[0:1,:]
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.iloc[0:nb_D,:]
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.iloc[0:nb_M,:]
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.iloc[0:nb_A,:]
                
                team_11=pd.concat([data_G,data_D,data_M,data_A])
                team_11=team_11.iloc[:,:-3]
                st.write("Le schéma qu'il aurait fallu choisir, un ",nb_D,"-",nb_M,"-",nb_A )
                st.write("Tu aurais pu compter sur ",team_11["nb_But"].sum(),"but(s) réel(s)")
                if nb_D==4 :
                    st.write("Bonus défense à 4: +0.5")
                elif nb_D==5 :
                    st.write("Bonus défense à 5: +1")
                st.write("Ton nombre de titulaires au coup d'envoi ",team_11["nb_tit"].sum())
                
                if team_11[team_11["Poste"]=="G"]["Note"].sum()<5 :
                    st.write(" Ton gardien était à la ramasse")
                elif team_11[team_11["Poste"]=="G"]["Note"].sum()>=8 :
                    st.write(" Ton gardien était chaud! Il annule un but")
                    
                st.write("Le nombre de joueurs vraiment bon (note>6.5):",team_11[(team_11["Note"]>=7)&(team_11["Poste"]!="G")]['nb_Note'].sum())
                st.write("dont :",team_11[(team_11["Note"]>=7)&(team_11["nb_But"]==0)&(team_11["Poste"]!="G")]['nb_Note'].sum(),"qui n'ont pas marqué en vrai. But virtuel?")
        
      
                
                def highlight_cells(val, color_if_true, color_if_false):
                    color = color_if_true if val == 1 else color_if_false
                    return 'background-color: {}'.format(color)
                #st.dataframe(team_11.style.applymap(highlight_cells, color_if_true='yellow', color_if_false='#C6E2E9',subset=['But/Match noté']))
                st.dataframe(team_11.style.format('{:.2f}', subset=['Note']).applymap(highlight_cells, color_if_true='yellow', color_if_false='white',subset=['nb_But']))
                #st.dataframe(team_11["But/Match Noté"].style.highlight_max())
                
                
            my_expander = st.expander(label="afficher l'équipe?",expanded=False)
            
            my_selection=team_11['Joueur'].values.tolist()
            data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')
            data=data[data['Joueur'].isin(my_selection)]
            #data=data.sort_values(by=['Note'],ascending=False)
            data.index = np.arange(1, len(data) + 1)
            #st.dataframe(data.round(2))
            #Pour afficher ou non, son équipe, selon le meilleur schéma tactique
            tactique_D,tactique_M,tactique_A=nb_D,nb_M,nb_A
            
             
            with my_expander:
                with st.container():#ligne gardien
                    col1, col2, col3,col4,col5 = st.columns(5)
                    local_G=data[data['Poste_bis']=='G']["Joueur"]
                    list_club=data[data['Poste_bis']=='G']["Club"]
                    with col3:
                        img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                        label=local_G.iloc[0]
                        st.image(img,caption=label,width=80)
                        
                with st.container(): #défense
                    local_D=data[data['Poste_bis']=='D']["Joueur"]
                    list_club=data[data['Poste_bis']=='D']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_D==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_D==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80)
                
                with st.container():#milieu
                    local_D=data[data['Poste_bis']=='M']["Joueur"]
                    list_club=data[data['Poste_bis']=='M']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_M==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_M==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80) 
                            
                with st.container():#attaque
                    local_D=data[data['Poste_bis']=='A']["Joueur"]
                    list_club=data[data['Poste_bis']=='A']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_A==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_A==1 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col3 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                            
            ##############################
            
            st.markdown("###  Préparer mon match")
            st.write("Quelle équipe d'après le modèle prédictif?")
            
            #Pour influencer la tactique du prochain match, (trier les résultats du modèle prédictif)
            st.write("Mais d'abord,le projet de jeu du coach")
            coaching=["Equilibré","Privilégier les Top joueurs (3)","Eviter les rotaldos (0)","Prendre une valise"]
            coaching_options = st.selectbox(
                 'Sélectionner le projet de jeu',
                 coaching)
            
            coef_1,coef_2,coef_3,coef_0=1,2,3,-1 #initialiser les différents coefficients
            if coaching_options=="Privilégier les Top joueurs (3)":
                coef_1,coef_2,coef_3,coef_0=0,0,1,0
            elif coaching_options=="Eviter les rotaldos (0)":
                coef_1,coef_2,coef_3,coef_0=0,0,0,-1
            elif coaching_options=="Prendre une valise":
                coef_1,coef_2,coef_3,coef_0=0,0,0,1
            else :
                coef_1,coef_2,coef_3,coef_0=1,2,3,-1
            
            #my_expander_prédictions = st.expander(label="afficher les prédictions pour le prochain match",expanded=True)
            
            pred1=pd.read_csv('data/Predictions_compo_1.csv')
            proba1=pd.read_csv("data/Probas_compo_1.csv")
            pred_proba1=pred1.merge(proba1, left_on='Joueur', right_on='Joueur')
            pred_proba1=pred_proba1.sort_values('Classe',ascending=False)
            pred_proba1.columns=['Joueur','Classe','0','1','2','3']
            pred_proba1["pred"]=pred_proba1.iloc[:,-4:].idxmax(axis=1).astype(int)
            pred_proba1["pred_cumul"]=coef_1*pred_proba1["1"]+coef_2*pred_proba1["2"]+coef_3*pred_proba1["3"]+coef_0*pred_proba1["0"]
            #pred_proba1["pred_cumul"]=1*pred_proba1["1"]+2*pred_proba1["2"]+3*pred_proba1["3"]-1*pred_proba1["0"]
            pred_proba1=pred_proba1.drop(columns=["pred"])
            pred_proba1=pred_proba1.sort_values('pred_cumul',ascending=False)
            
            #pred_proba1["pred_cumul"]=1*pred_proba1["Classe 1"]+2*pred_proba1["Classe 2"]+3*pred_proba1["Classe 3"]-1*pred_proba1["Classe 0"]
            my_expander_prédictions = st.expander(label="afficher les prédictions pour le prochain match",expanded=False)
            
            
            with my_expander_prédictions :
                st.dataframe(pred_proba1)    
            
            nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
            nb_team=0
            
            
            st.write("Un schéma préférentiel coach?")
            philo_jeu=["Oui, j'ai mes habitudes","Non, je m'adapte à l'adversaire"]
            schema_prefere=st.selectbox("Alors?",philo_jeu)
            
            if schema_prefere=="Oui, j'ai mes habitudes":
                tactique = st.selectbox(
                'Ok,laquelle?',
                ('3-4-3','3-5-2','4-3-3', '4-4-2','4-5-1','5-3-2','5-4-1'))
                #nb_D,nb_M,nb_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
                #nb_team=10
                #tactique_D,tactique_M,tactique_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
            
            my_expander_pred1 = st.expander(label="afficher la compo idéale?",expanded=True)
            
            with my_expander_pred1: 
                functions_to_apply = {
                    'Note_Jour' : ['mean'],
                    'Match_Noté':["count"],
                    'Titulaire_Match':['sum'],
                    'But_Match':['sum']
                }
                compo_1=pd.read_csv('data/Compo_1.csv',header=None)
                compo_1.columns=["Joueur"]
                df_compo_1=df0.loc[df0["Joueur"].isin(compo_1["Joueur"])]
                
                df_graph=df_compo_1[df_compo_1['Jour']==dernière_J]
                #df_graph=df_compo_1
                data =df_graph.groupby(['Club','Joueur','Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
                data.columns=['Club','Joueur','Poste','Note','nb_Note','nb_tit',"nb_But"]
                data=data.merge(pred_proba1,on='Joueur')
                data=data.drop(columns=['Note','nb_tit',"nb_But"])
                
                #data["But/Match noté"]=data["nb_But"]/data["nb_Note"]
                #data["But/Match noté"]=data["But/Match noté"].round(2)
                #data["%Titulaire_noté"]=data["nb_tit"]/data["nb_Note"]
                #data["%Titulaire_noté"]=data["%Titulaire_noté"].round(2)
                #data["%Titulaire"]=data["nb_tit"]/data["nb_Note"]
                #data["%Titulaire"]=data["%Titulaire"].round(2)
                data=data.fillna(0)
                data.sort_values(by=['pred_cumul'],ascending=False,inplace=True)
                #st.dataframe(data)
                
                #méthode itérative pour récupérer les positions 1 à 1 et en déduire le schéma tactique
                #nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
                #nb_team=0
                
                if schema_prefere=="Oui, j'ai mes habitudes":
                    nb_D,nb_M,nb_A=int(tactique[0]),int(tactique[2]),int(tactique[4])
                    nb_team=10
                #méthode itérative pour récupérer les positions 1 à 1 et en déduire le schéma tactique
                else:
                    nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
                    nb_team=0
                    for p in data["Poste"] :
                        if nb_team <=8 :
                            if (p=="D") & (nb_D<5):
                                nb_D +=1
                                nb_team +=1
                            elif (p=="M") & (nb_M<5):
                                nb_M +=1
                                nb_team +=1
                            elif (p=="A") & (nb_A<3):
                                nb_A +=1
                                nb_team +=1 
                                
                            #print(nb_D,nb_M,nb_A)
                        elif nb_team <10:
                            if (nb_D==4) and (nb_M==2):
                                nb_M +=1
                                nb_team +=1
                                
                            elif (nb_D==2) and (nb_M==5):
                                nb_D +=1
                                nb_team +=1
                            elif (nb_D==2) and (nb_M==4):
                                nb_D +=1
                                nb_team +=1
                            elif (nb_D==3) and (nb_M==3):
                                nb_D +=1
                                nb_team +=1
                            elif (nb_D==5) and (nb_M==2):
                                nb_M +=1
                                nb_team +=1
                            elif (nb_D==4) and (nb_M==4):
                                nb_A +=1
                                nb_team +=1
                            elif (nb_D==3) and (nb_M==4):
                                nb_D +=1
                                nb_team +=1    
                            elif (nb_D==5) and (nb_M==1):
                                nb_M =3
                                nb_D=4
                                nb_A=3
                                nb_team +=1    
                            elif (nb_D==5) and (nb_M==3):
                                nb_A +=1
                                nb_team +=1
                
                #si pas assez de défenseur, on applique un 4-4-2 par défaut
                if nb_D<3 :
                   nb_D=4
                   nb_M=4
                   nb_A=2
                
                #on récuprère les lignes de chaques postes
                        
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.head(1)
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.head(5)
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.head(5)
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.head(3)
                
                #on ne conserve que les premières lignes nécessaires
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.iloc[0:1,:]
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.iloc[0:nb_D,:]
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.iloc[0:nb_M,:]
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.iloc[0:nb_A,:]
                
                team_11=pd.concat([data_G,data_D,data_M,data_A])
                #team_11=team_11.iloc[:,:-3]
                st.write("Le schéma à choisir, un ",nb_D,"-",nb_M,"-",nb_A )
                #st.write("Tu aurais pu compter sur ",team_11["nb_But"].sum(),"but(s) réel(s)")
                if nb_D==4 :
                    st.write("Bonus défense à 4: +0.5")
                elif nb_D==5 :
                    st.write("Bonus défense à 5: +1")
                #st.write("Ton nombre de titulaires au coup d'envoi ",team_11["nb_tit"].sum())
                
                if team_11[team_11["Classe"]==0]["nb_Note"].sum()>=3 :
                    st.write(" Aïe, risque de CSC , trop de Rotaldos")
                elif team_11[team_11["Classe"]==0]["nb_Note"].sum()==2 :
                    st.write(" Attention , 1 rotaldo de plus et c'est le CSC")
                else :
                    st.write(" Normalement, pas de risque de CSC")
                    
                st.write("Le nombre de joueurs de classe 3:",team_11[(team_11["Classe"]==3)]['nb_Note'].sum())
                st.write("Le nombre de joueurs de classe 0 et 1:",team_11[(team_11["Classe"]<=1)]['nb_Note'].sum())
                #st.write("dont :",team_11[(team_11["Note"]>=7)&(team_11["nb_But"]==0)&(team_11["Poste"]!="G")]['nb_Note'].sum(),"qui n'ont pas marqué en vrai. But virtuel?")
        
      
                
                def highlight_cells(val, color_if_true, color_if_false):
                    color = color_if_true if val == 3 else color_if_false
                    return 'background-color: {}'.format(color)
                #st.dataframe(team_11.style.applymap(highlight_cells, color_if_true='yellow', color_if_false='#C6E2E9',subset=['But/Match noté']))
                st.dataframe(team_11.style.format('{:.2f}', subset=['Classe']).applymap(highlight_cells, color_if_true='yellow', color_if_false='white',subset=['Classe']))
                #st.dataframe(team_11["But/Match Noté"].style.highlight_max())
            
            
            my_expander_pred = st.expander(label="afficher l'équipe idéale?",expanded=False)
            
            my_selection=team_11['Joueur'].values.tolist()
            data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')
            data=data[data['Joueur'].isin(my_selection)]
            #data=data.sort_values(by=['Note'],ascending=False)
            data.index = np.arange(1, len(data) + 1)
            #st.dataframe(data.round(2))
            #Pour afficher ou non, son équipe, selon le meilleur schéma tactique
            tactique_D,tactique_M,tactique_A=nb_D,nb_M,nb_A
                
                 
            with my_expander_pred:
                with st.container():#ligne gardien
                    col1, col2, col3,col4,col5 = st.columns(5)
                    local_G=data[data['Poste_bis']=='G']["Joueur"]
                    list_club=data[data['Poste_bis']=='G']["Club"]
                    with col3:
                        img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                        label=local_G.iloc[0]
                        st.image(img,caption=label,width=80)
                        
                with st.container(): #défense
                    local_D=data[data['Poste_bis']=='D']["Joueur"]
                    list_club=data[data['Poste_bis']=='D']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_D==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_D==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80)
                
                with st.container():#milieu
                    local_D=data[data['Poste_bis']=='M']["Joueur"]
                    list_club=data[data['Poste_bis']=='M']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_M==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_M==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80) 
                            
                with st.container():#attaque
                    local_D=data[data['Poste_bis']=='A']["Joueur"]
                    list_club=data[data['Poste_bis']=='A']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_A==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_A==1 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col3 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                            
            ##############################
                
            st.markdown("###  Anticiper l'équipe adverse")
            st.write("Quelle équipe pour mon adversaire d'après le modèle prédictif?")
            
            adv=["Compo2","Compo3"]
            adv_options = st.selectbox(
                 'Sélectionner adversaire',
                 adv)
            
            my_expander_prédictions_adv = st.expander(label="afficher les prédictions pour le prochain match",expanded=False)
            
            
            if adv_options=="Compo2" :
                pred2=pd.read_csv('data/Predictions_compo_2.csv')
                proba2=pd.read_csv("data/Probas_compo_2.csv")
            else :
                pred2=pd.read_csv('data/Predictions_compo_3.csv')
                proba2=pd.read_csv("data/Probas_compo_3.csv")
            
            
            pred_proba2=pred2.merge(proba2, left_on='Joueur', right_on='Joueur')
            pred_proba2=pred_proba2.sort_values('Classe',ascending=False)
            pred_proba2.columns=['Joueur','Classe','0','1','2','3']
            pred_proba2["pred"]=pred_proba2.iloc[:,-4:].idxmax(axis=1).astype(int)
            pred_proba2["pred_cumul"]=1*pred_proba2["1"]+2*pred_proba2["2"]+3*pred_proba2["3"]-1*pred_proba2["0"]
            pred_proba2=pred_proba2.drop(columns=["pred"])
            
            #pred_proba1["pred_cumul"]=1*pred_proba1["Classe 1"]+2*pred_proba1["Classe 2"]+3*pred_proba1["Classe 3"]-1*pred_proba1["Classe 0"]
            
            
            
            with my_expander_prédictions_adv :
                st.dataframe(pred_proba2)    
            
            nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
            nb_team=0
            
            my_expander_pred2 = st.expander(label="afficher la compo qui représentera le plus grand défi?",expanded=False)
            
            with my_expander_pred2: 
                functions_to_apply = {
                    'Note_Jour' : ['mean'],
                    'Match_Noté':["count"],
                    'Titulaire_Match':['sum'],
                    'But_Match':['sum']
                }
                
                if adv_options=="Compo2" :
                    compo_2=pd.read_csv('data/Compo_2.csv',header=None)
                else :
                    compo_2=pd.read_csv('data/Compo_3.csv',header=None)
                
                compo_2.columns=["Joueur"]
                df_compo_2=df0.loc[df0["Joueur"].isin(compo_2["Joueur"])]
                
                df_graph=df_compo_2[df_compo_2['Jour']==dernière_J]
                #df_graph=df_compo_1
                data =df_graph.groupby(['Club','Joueur','Poste_bis'],as_index=False).agg(functions_to_apply).round(2)
                data.columns=['Club','Joueur','Poste','Note','nb_Note','nb_tit',"nb_But"]
                data=data.merge(pred_proba2,on='Joueur')
                data=data.drop(columns=['Note','nb_tit',"nb_But"])
                
                #data["But/Match noté"]=data["nb_But"]/data["nb_Note"]
                #data["But/Match noté"]=data["But/Match noté"].round(2)
                #data["%Titulaire_noté"]=data["nb_tit"]/data["nb_Note"]
                #data["%Titulaire_noté"]=data["%Titulaire_noté"].round(2)
                #data["%Titulaire"]=data["nb_tit"]/data["nb_Note"]
                #data["%Titulaire"]=data["%Titulaire"].round(2)
                data=data.fillna(0)
                data.sort_values(by=['pred_cumul'],ascending=False,inplace=True)
                #st.dataframe(data)
                
                #méthode itérative pour récupérer les positions 1 à 1 et en déduire le schéma tactique
                nb_G,nb_D,nb_M,nb_A=0,0,0,0 #initialiser les différents compteurs
                nb_team=0
    
                for p in data["Poste"] :
                    if nb_team <=8 :
                        if (p=="D") & (nb_D<5):
                            nb_D +=1
                            nb_team +=1
                        elif (p=="M") & (nb_M<5):
                            nb_M +=1
                            nb_team +=1
                        elif (p=="A") & (nb_A<3):
                            nb_A +=1
                            nb_team +=1 
                            
                        #print(nb_D,nb_M,nb_A)
                    elif nb_team <10:
                        if (nb_D==4) and (nb_M==2):
                            nb_M +=1
                            nb_team +=1
                            
                        elif (nb_D==2) and (nb_M==5):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==2) and (nb_M==4):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==3) and (nb_M==3):
                            nb_D +=1
                            nb_team +=1
                        elif (nb_D==5) and (nb_M==2):
                            nb_M +=1
                            nb_team +=1
                        elif (nb_D==4) and (nb_M==4):
                            nb_A +=1
                            nb_team +=1
                        elif (nb_D==3) and (nb_M==4):
                            nb_D +=1
                            nb_team +=1    
                        elif (nb_D==5) and (nb_M==1):
                            nb_M =3
                            nb_D=4
                            nb_A=3
                            nb_team +=1    
                        elif (nb_D==5) and (nb_M==3):
                            nb_A +=1
                            nb_team +=1
                                      
                #on récuprère les lignes de chaques postes
                        
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.head(1)
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.head(5)
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.head(5)
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.head(3)
                
                #on ne conserve que les premières lignes nécessaires
                data_G=data[data["Poste"]=="G"]
                data_G=data_G.iloc[0:1,:]
                data_D=data[data["Poste"]=="D"]
                data_D=data_D.iloc[0:nb_D,:]
                data_M=data[data["Poste"]=="M"]
                data_M=data_M.iloc[0:nb_M,:]
                data_A=data[data["Poste"]=="A"]
                data_A=data_A.iloc[0:nb_A,:]
                
                team_11=pd.concat([data_G,data_D,data_M,data_A])
                #team_11=team_11.iloc[:,:-3]
                st.write("Le schéma adverse probable, un ",nb_D,"-",nb_M,"-",nb_A )
                #st.write("Tu aurais pu compter sur ",team_11["nb_But"].sum(),"but(s) réel(s)")
                if nb_D==4 :
                    st.write("Bonus défense à 4: +0.5")
                elif nb_D==5 :
                    st.write("Bonus défense à 5: +1, çà va être plus compliqué de mettre un but virtuel")
                #st.write("Ton nombre de titulaires au coup d'envoi ",team_11["nb_tit"].sum())
                
                if team_11[team_11["Classe"]==0]["nb_Note"].sum()>=3 :
                    st.write(" Normalement, t'affrontes une équipe de Rotaldos")
                elif team_11[team_11["Classe"]==0]["nb_Note"].sum()==2 :
                    st.write("1 rotaldo de plus et c'est le CSC en ta faveur. Faut peut-être songer à un petit bonus.")
                else :
                    st.write(" Normalement, faut pas compter sur un CSC en ta faveur pour gagner le match.")
                    
                st.write("Le nombre de joueurs de classe 3:",team_11[(team_11["Classe"]==3)]['nb_Note'].sum())
                st.write("Le nombre de joueurs de classe 0 et 1:",team_11[(team_11["Classe"]<=1)]['nb_Note'].sum())
                #st.write("dont :",team_11[(team_11["Note"]>=7)&(team_11["nb_But"]==0)&(team_11["Poste"]!="G")]['nb_Note'].sum(),"qui n'ont pas marqué en vrai. But virtuel?")
        
      
                
                def highlight_cells(val, color_if_true, color_if_false):
                    color = color_if_true if val == 3 else color_if_false
                    return 'background-color: {}'.format(color)
                #st.dataframe(team_11.style.applymap(highlight_cells, color_if_true='yellow', color_if_false='#C6E2E9',subset=['But/Match noté']))
                st.dataframe(team_11.style.format('{:.2f}', subset=['Classe']).applymap(highlight_cells, color_if_true='red', color_if_false='white',subset=['Classe']))
                #st.dataframe(team_11["But/Match Noté"].style.highlight_max())
            
            
            my_expander_pred2 = st.expander(label="afficher l'équipe adverse la plus coriace?",expanded=False)
            
            my_selection=team_11['Joueur'].values.tolist()
            data=df_MPG_Current[['Joueur','Poste_bis','Club','Côte','Enchère moy','Note']]
            data=data.drop_duplicates(keep = 'first')
            data=data[data['Joueur'].isin(my_selection)]
            #data=data.sort_values(by=['Note'],ascending=False)
            data.index = np.arange(1, len(data) + 1)
            #st.dataframe(data.round(2))
            #Pour afficher ou non, son équipe, selon le meilleur schéma tactique
            tactique_D,tactique_M,tactique_A=nb_D,nb_M,nb_A
                
                 
            with my_expander_pred2:
                with st.container():#ligne gardien
                    col1, col2, col3,col4,col5 = st.columns(5)
                    local_G=data[data['Poste_bis']=='G']["Joueur"]
                    list_club=data[data['Poste_bis']=='G']["Club"]
                    with col3:
                        img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                        label=local_G.iloc[0]
                        st.image(img,caption=label,width=80)
                        
                with st.container(): #défense
                    local_D=data[data['Poste_bis']=='D']["Joueur"]
                    list_club=data[data['Poste_bis']=='D']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_D==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_D==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80)
                
                with st.container():#milieu
                    local_D=data[data['Poste_bis']=='M']["Joueur"]
                    list_club=data[data['Poste_bis']=='M']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_M==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_M==4:
                        col1, col2, col3,col4 = st.columns(4)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col1 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[3]+".png")
                            label=local_D.iloc[3]
                            st.image(img,caption=label,width=80) 
                        with col5 :
                            img=glob.glob(path_maillots+list_club.iloc[4]+".png")
                            label=local_D.iloc[4]
                            st.image(img,caption=label,width=80) 
                            
                with st.container():#attaque
                    local_D=data[data['Poste_bis']=='A']["Joueur"]
                    list_club=data[data['Poste_bis']=='A']["Club"]
                    #l=glob.glob("Maillots/"+list_club+".png")
                    if tactique_A==3 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col3 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[2]+".png")
                            label=local_D.iloc[2]
                            st.image(img,caption=label,width=80)    
                    elif tactique_A==1 :
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col3 :
                            #st.write(local_D.iloc[0])
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                    else:
                        col1, col2, col3,col4,col5 = st.columns(5)
                        with col2 :
                            img=glob.glob(path_maillots+list_club.iloc[0]+".png")
                            label=local_D.iloc[0]
                            st.image(img,caption=label,width=80)
                        with col4 :
                            img=glob.glob(path_maillots+list_club.iloc[1]+".png")
                            label=local_D.iloc[1]
                            st.image(img,caption=label,width=80)
                            
                            
            ##############################   
             
            st.success("On siffle la fin de match, il n'y a plus qu'à lancer une nouvelle compétition MPG pour tester pour de vrai!")
