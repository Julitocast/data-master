	# Graph des notes jour

	fig1 = sns.displot(df1['Note_Jour'], bins=25, color="#87E377")
	plt.xlabel("Note à l'issue de la journée")
	plt.ylabel("Nombre de notes")
	st.pyplot(fig1)


	# Aperçu de la répartition de la variable pour estimer les classes

	col1, col2 = st.columns([4,4])

	with col1:
		st.markdown("<p style='text-align: center; '>Répartition de la variable 'Note_Jour'</p>", unsafe_allow_html=True)

		df_graph=df[df['Note'] != 0]
		fig1 = sns.displot(df_graph['Note_Jour'], bins=25, color="#87E377")
		plt.annotate("Joueurs n'ayant pas joué\n la dernière journée",
		             xy=(0.5, 5000),
		             xytext=(3, 4800),
		             arrowprops={'facecolor':'green'})
		plt.xticks(np.arange(0,9))
		plt.xlabel("Note à l'issue de la journée")
		plt.ylabel("Nombre de notes")
		st.pyplot(fig1, figsize=(5,5))

	with col2:
		st.markdown("<p style='text-align: center; '>Focus sur les notes différentes de 0</p>", unsafe_allow_html=True)

		fig2 = sns.displot(df_graph['Note_Jour'][df_graph['Note_Jour'] !=0], bins=25, color="#87E377")
		plt.xticks(np.arange(0,9))
		plt.xlabel("Note à l'issue de la journée")
		plt.ylabel("Nombre de notes")
		st.pyplot(fig2, figsize=(5,5))


	# Code Python pour le graphique
	df_graph=df[df['Note_Jour']!=0]
	fig = sns.displot(data=df_graph,
 	           x='Note_Jour',
	            col='Poste',
	            hue='Poste',
	            col_wrap=3,
	            kde=True,
	            palette='CMRmap',
	            col_order=["G", "DC", "DL", "MD", "MO", "A"],
	            bins=25,
	            legend=False)
	st.pyplot(fig, figsize=(5,5))
	plt.xlim([4,6.5])
	plt.ylabel('Nombre de notes')
	plt.suptitle('Distribution des notes selon le poste',  y = 1.05, fontsize = 20, color='#001EB4')
	plt.show();