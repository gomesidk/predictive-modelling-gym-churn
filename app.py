import pandas as pd
import streamlit as st
import pickle
import os
from pathlib import Path


st.set_page_config(page_title="ML Trainer", layout="wide")

st.title("Web App de Treino de Modelos")

MODEL_PATH = Path(__file__).parent / "final_xgb_model.pkl"


@st.cache_resource
def load_model():
	"""Carrega o modelo pkl uma vez e cachea."""
	if not MODEL_PATH.exists():
		st.error(f"Modelo não encontrado: {MODEL_PATH}")
		return None
	try:
		with open(MODEL_PATH, "rb") as f:
			return pickle.load(f)
	except Exception as e:
		st.error(f"Erro ao carregar o modelo: {e}")
		return None


tab1, tab2 = st.tabs(["Upload Dataset", "Previsões"])

with tab1:
	st.write("Faz upload do teu dataset CSV.")

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


with tab2:
	st.write("Faz previsões usando o modelo treinado.")

	if "dataset" not in st.session_state:
		st.warning("Primeiro, faz upload de um dataset na aba anterior.")
	else:
		model = load_model()

		if model is None:
			st.error("Não foi possível carregar o modelo.")
		else:
			st.success("Modelo carregado com sucesso!")

			st.subheader("Configuração")
			threshold = st.slider(
				"Threshold de risco (0-100%)",
				min_value=0,
				max_value=100,
				value=50,
				help="Valor mínimo para considerar como risco elevado.",
			)

			try:
				df = st.session_state["dataset"].copy()

				df.columns = df.columns.str.lower()

				if "churn" in df.columns:
					df = df.drop(columns=["churn"])

				expected_features = [
					'gender', 'near_location', 'partner', 'promo_friends', 'phone',
					'contract_period', 'group_visits', 'age', 'avg_additional_charges_total',
					'month_to_end_contract', 'lifetime', 'avg_class_frequency_total',
					'avg_class_frequency_current_month'
				]

				missing_cols = [col for col in expected_features if col not in df.columns]
				if missing_cols:
					st.error(f"Colunas em falta no dataset: {', '.join(missing_cols)}")
				else:
					df_model = df[expected_features]

					try:
						predictions = model.predict_proba(df_model)
						risk_scores = predictions[:, 1] * 100
					except AttributeError:
						predictions = model.predict(df_model)
						risk_scores = predictions * 100

					df["Risco de Desistência (%)"] = risk_scores.round(2)

					df_sorted = df.sort_values("Risco de Desistência (%)", ascending=False).reset_index(drop=True)

					df_sorted["Classificação"] = df_sorted["Risco de Desistência (%)"].apply(
						lambda x: "Muito Alto" if x >= threshold
						else "Médio" if x >= threshold * 0.6
						else "Baixo"
					)

					st.subheader("Resultados das Previsões")

					metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
					with metrics_col1:
						high_risk = (df_sorted["Risco de Desistência (%)"] >= threshold).sum()
						st.metric("Alto Risco", high_risk)

					with metrics_col2:
						avg_risk = df_sorted["Risco de Desistência (%)"].mean()
						st.metric("Risco Médio", f"{avg_risk:.1f}%")

					with metrics_col3:
						total_records = len(df_sorted)
						st.metric("Total de Registos", total_records)

					st.dataframe(
						df_sorted,
						use_container_width=True,
						hide_index=False,
					)

					csv = df_sorted.to_csv(index=False)
					st.download_button(
						label="⬇️ Download Resultados (CSV)",
						data=csv,
						file_name="predicoes_risco_desistencia.csv",
						mime="text/csv",
					)

			except Exception as error:
				st.error("Erro ao processar o modelo ou fazer previsões.")
				st.exception(error)
