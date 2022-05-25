# Commandes pour lancer le fichier

# cd C:\Users\HP\Documents\DATA\Projet MPG\STREAMLIT
# streamlit run Streamlit_MPG_Julien.py

import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
#st.set_page_config(layout="wide")

header = st.container()
intro = st.container()
dataset = st.container()
dataviz = st.container()
model_training = st.container()
final = st.container()

df = pd.read_csv('data/MPG_df_dataviz.csv', sep=',') # DF pour la partie Dataviz
df0 = pd.read_csv('data/MPG_Current_5j.csv', sep=',') # DF pour la partie Dataviz
df1 = pd.read_csv('data/MPG_df_final_ML.csv', sep=',', index_col='Unnamed: 0')

# Création des DF avec les notes différentes de 0
df_00      = df[df['Note'] != 0].sort_values('Journée', ascending=True) # DF global sans les notes moyennes à 0 (joueurs qui ne jouent jamais)
df_00_jour = df[df['Note_Jour']!=0].sort_values('Journée', ascending=True) # DF sans les notes journées à 0

# Sidebar menu

# navigation = ["Dataset", "Dataviz", "Machine learning"]
# choice = st.sidebar.selectbox("Navigation", navigation)

# Autre sidebar

# with st.sidebar:
# 	choice = option_menu(menu_title = "Navigation",
# 						options = ["Intro", "Dataset", "Dataviz", "Machine learning", "Tests réels"],
# 						icons = ["caret-right-square-fill", "server", "file-bar-graph-fill", "fan", "trophy-fill"],
# 						menu_icon = "bookmark-star",
# 						default_index = 0,
# 						)



########################################################
# 0/ HEADER
########################################################

with header:
	st.markdown("<h1 style='text-align: center; color: grey;'>&#9917 Projet <span style='color: #00B727;'>MPG</span> &#9917</h1>", unsafe_allow_html=True)

# Menu horizontal
	choice = option_menu(menu_title = None,
						options = ["Intro", "Dataset", "Dataviz", "Machine learning", "Tests réels"],
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

########################################################
# 0/ INTRO
########################################################

if choice == "Intro":
	with intro:
		st.markdown("<h3 style='text-align: center; color: grey;'>Le meilleur moyen de faire votre compo d'avant-match &#128076</h3><br>", unsafe_allow_html=True)
		#st.image("https://media.giphy.com/media/RJscCVJO5TvIskoCCR/giphy.gif")
		col1, col2, col3 = st.columns(3)
		with col1:
		    st.write("")
		with col2:
		    st.image("https://media.giphy.com/media/RJscCVJO5TvIskoCCR/giphy.gif")
		with col3:
		    st.write("")

		st.markdown("<p style='text-align: center; color: black;'>Ne choisissez plus votre équipe au hasard : analysez les performances de vos joueurs grâce au modèle de prédiction XXXX. <br> Et alignez la meilleure équipe pour affronter votre adversaire au prochain match.</p><br>", unsafe_allow_html=True)
		st.success("A vous la victoire !") 



########################################################
# 1/ DATASET
########################################################

elif choice == "Dataset":

	with dataset:
		st.header('1/ Récupération des données')

		st.markdown("*  **Problématique** / _Objectif_")
		st.markdown("*  Sources de données utilisées")
		st.markdown("*  Travail individuel de chacun des fichiers")
		st.markdown("*  Mutualisation et préparation du dataframe final")



########################################################
# 2/ DATAVIZ
########################################################

elif choice == "Dataviz":
	with dataviz:
		st.header('2/ **Visualisation des données**')

		show = st.checkbox("Cochez cette case si vous souhaitez afficher l'extrait du dataset final")
		if show:
			st.markdown("###### Voici l'extrait du dataframe final que nous allons utiliser pour la visualisation des données.")
			st.write(df.head())

		st.write("Au final, nous obtenons un dataframe avec ", df.shape[1], "colonnes.")
		st.write("* dont ", len(df.select_dtypes('number').columns), "variables **numériques**")
		st.write("* et ", len(df.select_dtypes('object').columns), "variables **catégorielles**")
		st.write("L'objectif de cette section est de présenter divers graphiques réprésentant la distribution de certaines variables clés et les relations qu'elles entretiennent entre elles.")

		
		########## 1ère partie Dataviz


		st.markdown("### :arrow_forward: Analyse de la variable cible : la Note du joueur")

		# 1/ Distribution de la note à l'issue d'une journée

		st.markdown("##### 1/ Distribution de la note à l'issue d'une journée")

		notes_liste = sorted(df_00['Note_Jour'].unique().tolist())
		start_note, end_note = st.select_slider("> Sélectionnez les notes puis activez la lecture pour voir l'évolution des notes au cours des journées", options = notes_liste, value=[0.0,9.0])
		df1 = df_00[(df_00['Note_Jour'] >= start_note) & (df_00['Note_Jour'] <= end_note)]

		fig1 = px.histogram(df1, x = 'Note_Jour',
							nbins = 25,
							labels = {'Note_Jour':'Note de la journée'},
							animation_frame = "Journée",
							color_discrete_sequence=['#00B727'],
							width = 800
							)
		fig1.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
		fig1.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 50
		fig1.update_layout(bargap = 0.1)
		st.write(fig1)

		st.markdown("Distribution globale de la note à l'issue d'une journée, sans les notes à 0")

		fig2 = px.histogram(df_00_jour, x = 'Note_Jour',
							nbins = 25,
							labels = {'Note_Jour':'Note de la journée'},
							color_discrete_sequence=['#00B727'],
							width = 600
							)
		fig2.update_layout(bargap = 0.1)
		st.write(fig2)


		# 2/ Répartition des notes par match selon le résultat du match

		st.markdown("##### 2/ Répartition des notes par match selon le résultat du match")
		st.image("media/charts/1-2-Répartition des notes par match selon le résultat du match.png")
		
		# 3/ Variation de la note selon le nombre de matchs joués

		st.markdown("##### 3/ Variation de la note selon le nombre de matchs joués")
		st.image("media/charts/1-3-Variation de la note selon le nombre de matchs joués.png")


		########## 2e partie Dataviz

		st.markdown("### :arrow_forward: Analyse de la note selon le poste du joueur")

		# 1/ Distribution de la note selon le poste

		st.markdown("##### 1/ Distribution de la note selon le poste")
		st.image("media/charts/2-1-Distribution des notes selon le poste.png")

		# 2/ Evolution de la note en fonction du nombre de buts (sauf Gardiens)

		st.markdown("##### 2/ Evolution de la note en fonction du nombre de buts (sauf Gardiens)")

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
		st.image("media/charts/2-3-Note par club selon poste.jpg")

		# 4/ Rôle du joueur dans le match, selon le poste

		st.markdown("##### 4/ Rôle du joueur dans le match, selon le poste")
		st.image("media/charts/2-4-Rôle du joueur dans le match pas type de poste.png")

		# 5/ Répartition des notes par club, selon le rôle du joueur

		st.markdown("##### 5/ Répartition des notes par club, selon le rôle du joueur")
		st.image("media/charts/2-5-Répartition des notes par club, selon le rôle du joueur.png")


		########## 3e partie Dataviz

		st.markdown("### :arrow_forward: Impact du type de déplacement de l'équipe")

		# 1/ Résultats des matchs selon le déplacement des équipes

		st.markdown("##### 1/ Résultats des matchs selon le déplacement des équipes")
		st.image("media/charts/3-1-Résultats des matchs selon le déplacement des équipes.png")

		# 2/ Nombre de buts marqués par Club selon le type de déplacement

		st.markdown("##### 2/ Nombre de buts marqués par Club selon le type de déplacement")
		st.image("media/charts/3-2-Nombre de buts marqués par Club selon le type de déplacement.png")

		# 3/ Note des joueurs selon le poste et le déplacement de l'équipe

		st.markdown("##### 3/ Note des joueurs selon le poste et le déplacement de l'équipe")
		st.image("media/charts/3-3-Note des joueurs selon le poste et le déplacement de l'équipe.png")



########################################################
# 3/ ENTRAINEMENTS DES MODELES DE ML
########################################################

elif choice == "Machine learning":
	with model_training:
		st.header('3/ Machine Learning')
		
		show = st.checkbox("Idem, si vous souhaitez afficher le dataset original, cochez la case")
		if show:
			st.markdown("###### Voici l'extrait du dataframe original, contenant les 94 colonnes.")
			st.write(df.head())

		st.write("Cette section se divise en 4 grandes parties :")
		st.write("* Choix des variables")
		st.write("* Entraînements des modèles")
		st.write("* Features engineering sur le modèle retenu")
		st.write("* ACP et conclusions")

		st.markdown("#### a/ Choix des variables")

		st.write("La première étape a été de réduire le nombre de variables pour le machine learning. Avec 94 variables, nous avons choisi de créer 3 matrices de corrélation pour plus de simplicité de lecture et d'analyse.")
		st.write("Voici les 3 matrices (à agrandir pour voir les détails) de corrélation avec la variable cible **'Note_Jour'** :")

		col1, col2, col3 = st.columns(3)
		with col1:
		    st.image("media/charts/Matrice de corrélation_01.png")
		with col2:
		    st.image("media/charts/Matrice de corrélation_02.png")
		with col3:
		    st.image("media/charts/Matrice de corrélation_03.png")

		st.write("Puis, nous avons effectué plusieurs ajustements sur les variables afin de ne conserver que celles qui étaient intéressantes pour l'entraînement des modèles.")

		st.write("**Forme du DF initial :**", df.shape)

		df.index = df['Joueur'] + '-' + df['Poste']
		df.drop(['Joueur', 'Poste'], axis=1, inplace = True)
		if st.button("> Remplacer l'index par une concaténation des variables 'Joueur' et 'Poste'"):
			st.write("Nouvelle forme du DF :", df.shape)

		to_keep = ['Côte', 'Enchère_moy', 'Note', 'Variation', 'But', 'Temps', 'Note_Jour', 'Statut', 'Jour',
           'Derniers_Résultats_avantJ', 'Rang_début_Journée', 'Rang_début_Journée_adv', 'Ecart_Pos_NumTag',
           'Derniers_Status_Match_avant_J', 'Derniers_Match_Buteur_avant_J', 'Derniers_But_Match_avant_J',
           '%Buteur_Derniers_Match', '%Titulaire_Derniers_Match','%Noté_Derniers_Match', 'But/Match_Derniers_Match',
           'Note_J_Moy_Poste_avantJ', 'Note_J_Moy_A_avantJ', 'Note_J_Moy_DC_avantJ', 'Note_J_Moy_DL_avantJ',
           'Note_J_Moy_G_avantJ', 'Note_J_Moy_MD_avantJ', 'Note_J_Moy_MO_avantJ', 'Note_Moy_Derniers_Match',
           'Note_Moy_Derniers_Match_Dom', 'Note_Moy_Derniers_Match_Ext','Moy_But_Av_J', 'Nb_But_Av_J', 'Rolling_Note_J',
           'Rolling_Note_J_Derniers_Match']
		df = df[to_keep]
		if st.button("> Exclure les variables les moins corrélées et les moins intéressantes d'un point de vue métier"):
			st.write("Nouvelle forme du DF :", df.shape)

		df = df[df['Note'] != 0]
		if st.button("> Retirer les lignes des joueurs avec une note moyenne à 0 (ie les joueurs qui ne jouent jamais)"):
			st.write("Nouvelle forme du DF :", df.shape)

		df = df[df['Jour']>5]
		if st.button("> Exclure les 5 premières journées de la saison car pas assez de data pour appliquer le modèle"):
			st.write("Nouvelle forme du DF :", df.shape)

		df['Ratio_G_derniers matchs'] = df['Derniers_Résultats_avantJ'].apply(lambda x: x.count('G')/len(x))
		df['Ratio_N_derniers matchs'] = df['Derniers_Résultats_avantJ'].apply(lambda x: x.count('N')/len(x))
		df['Ratio_P_derniers matchs'] = df['Derniers_Résultats_avantJ'].apply(lambda x: x.count('P')/len(x))
		df = df.drop('Derniers_Résultats_avantJ',axis=1)
		if st.button("> Numériser la variable indiquant la série des derniers résultats"):
			st.write("Nouvelle forme du DF :", df.shape)

		df['Statut'] = df['Statut'].replace({'Extérieur':0, 'Domicile':1})
		if st.button("> Numériser la variable indiquant le type de déplacement de l'équipe"):
			st.write("Nouvelle forme du DF :", df.shape)

		
		st.write("Enfin, nous avons créé les différentes classes de notre variable cible en la discrétisant.")


		st.write("**Forme du DF final :**", df.shape)

		st.markdown("#### b/ Entraînements des modèles")




		st.markdown("#### c/ Features engineering sur le modèle retenu")




		st.markdown("#### d/ ACP et conclusions")




########################################################
# 4/ TESTS REELS
########################################################

elif choice == "Tests réels":
	with model_training:
		st.header('4/ Tests réels')
