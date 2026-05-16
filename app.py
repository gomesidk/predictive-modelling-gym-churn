import pandas as pd
import streamlit as st


st.set_page_config(page_title="ML Trainer", page_icon="", layout="wide")

st.title("Web App de Treino de Modelos")
st.write("Começamos pela primeira funcionalidade: upload de dataset CSV.")

uploaded_file = st.file_uploader(
	"Faz upload do teu dataset (.csv)",
	type=["csv"],
	help="Só são aceites ficheiros CSV nesta fase.",
)

if uploaded_file is not None:
	try:
		df = pd.read_csv(uploaded_file)

		st.session_state["dataset"] = df

		st.success("Dataset carregado com sucesso!")

		col1, col2 = st.columns(2)
		col1.metric("Linhas", f"{df.shape[0]:,}".replace(",", "."))
		col2.metric("Colunas", df.shape[1])

		st.subheader("Pré-visualização do dataset")
		st.dataframe(df.head(20), use_container_width=True)

		with st.expander("Ver tipos de dados por coluna"):
			dtypes_df = pd.DataFrame(
				{
					"coluna": df.columns,
					"tipo": df.dtypes.astype(str).values,
				}
			)
			st.dataframe(dtypes_df, use_container_width=True)

	except Exception as error:
		st.error(
			"Não foi possível ler o ficheiro CSV. "
			"Confirma o formato e tenta novamente."
		)
		st.exception(error)
else:
	st.info("Aguardando upload de um ficheiro CSV.")
