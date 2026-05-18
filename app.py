import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


st.set_page_config(page_title="ML Trainer", layout="wide")

st.title("Web App de Treino de Modelos")

MODEL_PATH = Path(__file__).parent / "final_xgb_model.pkl"

EXPECTED_FEATURES = [
	"gender", "near_location", "partner", "promo_friends", "phone",
	"contract_period", "group_visits", "age", "avg_additional_charges_total",
	"month_to_end_contract", "lifetime", "avg_class_frequency_total",
	"avg_class_frequency_current_month",
]


@st.cache_resource
def load_model():
	"""Carrega o modelo pkl uma vez e cachea."""
	if not MODEL_PATH.exists():
		st.error(f"Modelo não encontrado: {MODEL_PATH}")
		return None
	try:
		with open(MODEL_PATH, "rb") as f:
			return joblib.load(f)
	except Exception as e:
		st.error(f"Erro ao carregar o modelo: {e}")
		return None
	
@st.cache_resource
def load_scaler():
	"""Carrega o scaler pkl uma vez e cachea."""
	scaler_path = Path(__file__).parent / "scaler.pkl"
	if not scaler_path.exists():
		st.error(f"Scaler não encontrado: {scaler_path}")
		return None
	try:
		with open(scaler_path, "rb") as f:
			return joblib.load(f)
	except Exception as e:
		st.error(f"Erro ao carregar o scaler: {e}")
		return None


def classify_risk(risk_value: float, threshold: float) -> str:
	if risk_value >= threshold:
		return "Muito Alto"
	if risk_value >= threshold * 0.6:
		return "Médio"
	return "Baixo"


tab1, tab2, tab3 = st.tabs(["Upload Dataset", "Previsões", "Simulador What-If"])

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
		scaler = load_scaler()

		if model is None or scaler is None:
			st.error("Não foi possível carregar o modelo e/ou scaler.")
		else:
			st.success("Modelo e scaler carregados com sucesso!")

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

				missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
				if missing_cols:
					st.error(f"Colunas em falta no dataset: {', '.join(missing_cols)}")
				else:
					df_model = df[EXPECTED_FEATURES]

					df_model_scaled = scaler.transform(df_model)
					if not isinstance(df_model_scaled, pd.DataFrame):
						df_model_scaled = pd.DataFrame(
							df_model_scaled,
							columns=EXPECTED_FEATURES,
							index=df_model.index,
						)

					try:
						predictions = model.predict_proba(df_model_scaled)
						risk_scores = predictions[:, 1] * 100
					except AttributeError:
						predictions = model.predict(df_model_scaled)
						risk_scores = predictions * 100

					df["Risco de Desistência (%)"] = risk_scores.round(2)

					df_sorted = df.sort_values("Risco de Desistência (%)", ascending=False).reset_index(drop=True)

					df_sorted["Classificação"] = df_sorted["Risco de Desistência (%)"].apply(
						lambda x: classify_risk(x, threshold)
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


with tab3:
	st.write("Simula um cliente hipotético e vê o risco de churn em tempo real.")

	model = load_model()
	scaler = load_scaler()

	if model is None or scaler is None:
		st.error("Não foi possível carregar o modelo e/ou scaler.")
	else:
		defaults = {
			"gender": 1,
			"near_location": 1,
			"partner": 0,
			"promo_friends": 0,
			"phone": 1,
			"contract_period": 6,
			"group_visits": 0,
			"age": 30,
			"avg_additional_charges_total": 150.0,
			"month_to_end_contract": 3,
			"lifetime": 4,
			"avg_class_frequency_total": 1.8,
			"avg_class_frequency_current_month": 1.8,
		}

		ranges = {
			"age": (18, 80),
			"avg_class_frequency_current_month": (0.0, 7.0),
			"contract_period": (1, 24),
		}

		if "dataset" in st.session_state:
			df_defaults = st.session_state["dataset"].copy()
			df_defaults.columns = df_defaults.columns.str.lower()
			if "churn" in df_defaults.columns:
				df_defaults = df_defaults.drop(columns=["churn"])

			if all(col in df_defaults.columns for col in EXPECTED_FEATURES):
				for col in EXPECTED_FEATURES:
					if pd.api.types.is_numeric_dtype(df_defaults[col]):
						defaults[col] = float(df_defaults[col].median())
					else:
						defaults[col] = float(df_defaults[col].mode().iloc[0])

				ranges["age"] = (
					int(df_defaults["age"].min()),
					int(df_defaults["age"].max()),
				)
				ranges["avg_class_frequency_current_month"] = (
					float(df_defaults["avg_class_frequency_current_month"].min()),
					float(df_defaults["avg_class_frequency_current_month"].max()),
				)
				ranges["contract_period"] = (
					int(df_defaults["contract_period"].min()),
					int(df_defaults["contract_period"].max()),
				)

		st.subheader("Parâmetros principais")
		col_a, col_b, col_c = st.columns(3)

		age = col_a.slider(
			"Idade",
			min_value=ranges["age"][0],
			max_value=ranges["age"][1],
			value=int(round(defaults["age"])),
		)
		freq_week = col_b.slider(
			"Frequência Semanal (média)",
			min_value=float(ranges["avg_class_frequency_current_month"][0]),
			max_value=float(ranges["avg_class_frequency_current_month"][1]),
			value=float(defaults["avg_class_frequency_current_month"]),
			step=0.1,
		)
		contract_months = col_c.slider(
			"Meses de Contrato",
			min_value=ranges["contract_period"][0],
			max_value=ranges["contract_period"][1],
			value=int(round(defaults["contract_period"])),
		)

		with st.expander("Parâmetros avançados"):
			adv1, adv2, adv3 = st.columns(3)

			gender = adv1.selectbox("Género", options=[0, 1], index=int(round(defaults["gender"])))
			near_location = adv2.selectbox("Perto do ginásio", options=[0, 1], index=int(round(defaults["near_location"])))
			partner = adv3.selectbox("Parceiro", options=[0, 1], index=int(round(defaults["partner"])))

			adv4, adv5, adv6 = st.columns(3)
			promo_friends = adv4.selectbox("Promo Friends", options=[0, 1], index=int(round(defaults["promo_friends"])))
			phone = adv5.selectbox("Telefone", options=[0, 1], index=int(round(defaults["phone"])))
			group_visits = adv6.selectbox("Visitas em grupo", options=[0, 1], index=int(round(defaults["group_visits"])))

			adv7, adv8, adv9, adv10 = st.columns(4)
			month_to_end_contract = adv7.number_input("Meses até fim contrato", min_value=0, max_value=36, value=int(round(defaults["month_to_end_contract"])))
			lifetime = adv8.number_input("Lifetime (meses)", min_value=0, max_value=72, value=int(round(defaults["lifetime"])))
			avg_additional_charges_total = adv9.number_input("Gastos adicionais médios", min_value=0.0, value=float(defaults["avg_additional_charges_total"]), step=1.0)
			avg_class_frequency_total = adv10.number_input("Frequência média total", min_value=0.0, value=float(defaults["avg_class_frequency_total"]), step=0.1)

		input_data = {
			"gender": int(gender),
			"near_location": int(near_location),
			"partner": int(partner),
			"promo_friends": int(promo_friends),
			"phone": int(phone),
			"contract_period": int(contract_months),
			"group_visits": int(group_visits),
			"age": float(age),
			"avg_additional_charges_total": float(avg_additional_charges_total),
			"month_to_end_contract": float(month_to_end_contract),
			"lifetime": float(lifetime),
			"avg_class_frequency_total": float(avg_class_frequency_total),
			"avg_class_frequency_current_month": float(freq_week),
		}

		df_one = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
		df_one_scaled = scaler.transform(df_one)
		if not isinstance(df_one_scaled, pd.DataFrame):
			df_one_scaled = pd.DataFrame(df_one_scaled, columns=EXPECTED_FEATURES)

		try:
			pred_one = model.predict_proba(df_one_scaled)
			risk_one = float(pred_one[0, 1]) * 100
		except AttributeError:
			pred_one = model.predict(df_one_scaled)
			risk_one = float(pred_one[0]) * 100

		st.subheader("Resultado em tempo real")
		res1, res2 = st.columns(2)
		res1.metric("Probabilidade de Churn", f"{risk_one:.2f}%")
		res2.metric("Classificação", classify_risk(risk_one, 50))
		st.progress(min(max(risk_one / 100.0, 0.0), 1.0))
