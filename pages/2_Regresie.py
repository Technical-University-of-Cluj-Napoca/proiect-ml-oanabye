import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, KFold

st.set_page_config(page_title='Regresie — Mașini', page_icon='🚗', layout='wide')

# ── Încărcare modele și date ──────────────────────────
@st.cache_resource
def load_models():
    models  = joblib.load('models/reg_models.pkl')
    scaler  = joblib.load('models/reg_scaler.pkl')
    features = joblib.load('models/reg_feature_names.pkl')
    return models, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv('data/second_hand_cars.csv')

models, scaler, feature_names = load_models()
df = load_data()

# ── Header ────────────────────────────────────────────
st.title('🚗 Predicția Prețului Mașinilor Second Hand')
st.markdown('''
> *"Data viitoare când cineva îți spune că mașina a fost condusă doar duminica
> la biserică — avem cu ce să îl contrazic cu date."*
''')
st.markdown('---')

# ── Tabs principale ───────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    '📊 Explorare Date',
    '🤖 Modele & Predicție',
    '🔍 Explicabilitate SHAP'
])

# ════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════
with tab1:
    st.header('Analiza Exploratorie a Datelor')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Număr mașini', f'{len(df):,}')
    col2.metric('Features', '10')
    col3.metric('Preț mediu', f'{df["current price"].mean():,.0f} INR')
    col4.metric('Preț maxim', f'{df["current price"].max():,.0f} INR')

    st.markdown('---')

    # Distribuția targetului
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Distribuția prețului')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['current price'], bins=30,
                color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Preț (INR)')
        ax.set_ylabel('Frecvență')
        ax.set_title('Distribuția current price')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader('Corelații cu prețul')
        fig, ax = plt.subplots(figsize=(7, 4))
        corr = df.drop(columns=['v.id']).corr()['current price'].drop('current price').sort_values()
        colors = ['coral' if x < 0 else 'steelblue' for x in corr]
        corr.plot(kind='barh', color=colors, ax=ax)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Corelația features cu prețul')
        st.pyplot(fig)
        plt.close()

    # Heatmap
    st.subheader('Matricea de Corelații')
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(df.drop(columns=['v.id']).corr(), dtype=bool))
    sns.heatmap(df.drop(columns=['v.id']).corr(), mask=mask,
                annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    st.pyplot(fig)
    plt.close()

    # Scatter plots
    st.subheader('Relația Features vs Preț')
    features_plot = ['on road old', 'on road now', 'years',
                     'km', 'hp', 'condition']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(features_plot):
        axes[i].scatter(df[col], df['current price'],
                        alpha=0.4, color='steelblue', s=15)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Preț')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════
# TAB 2 — MODELE & PREDICȚIE
# ════════════════════════════════════════════════════
with tab2:
    st.header('Modele & Predicție')

    # ── Selectare model ───────────────────────────────
    model_name = st.selectbox(
        '🔽 Selectează modelul:',
        list(models.keys())
    )
    model = models[model_name]

    st.markdown('---')

    col1, col2 = st.columns([1, 1])

    # ── Hiperparametri ────────────────────────────────
    with col1:
        st.subheader('⚙️ Hiperparametri')
        params = model.get_params()
        params_df = pd.DataFrame(
            list(params.items()),
            columns=['Parametru', 'Valoare']
        )
        st.dataframe(params_df, use_container_width=True)

    # ── Metrici ───────────────────────────────────────
    with col2:
        st.subheader('📈 Metrici pe setul de test')

        # Recalculează pe test
        X_reg = df.drop(columns=['v.id', 'current price'])
        y_reg = df['current price']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.25, random_state=42
        )
        X_test_sc = scaler.transform(X_test)
        y_pred    = model.predict(X_test_sc)

        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        m1, m2 = st.columns(2)
        m1.metric('R²',   f'{r2:.4f}')
        m2.metric('RMSE', f'{rmse:,.0f} INR')
        m3, m4 = st.columns(2)
        m3.metric('MAE',  f'{mae:,.0f} INR')
        m4.metric('MSE',  f'{mse:,.0f}')

    st.markdown('---')

    # ── Predicții Reale vs Prezise ─────────────────────
    st.subheader('Predicții Reale vs Prezise')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Ideal')
    ax.set_xlabel('Preț Real')
    ax.set_ylabel('Preț Prezis')
    ax.set_title(f'Real vs Prezis — {model_name}')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # ── Curbe de învățare ─────────────────────────────
    st.subheader('📉 Curbe de Învățare')
    with st.spinner('Se calculează curbele de învățare...'):
        X_train_sc = scaler.transform(X_train)
        cv_lc = KFold(n_splits=3, shuffle=True, random_state=42)

        train_sizes, train_sc, val_sc = learning_curve(
            model, X_train_sc, y_train,
            train_sizes=np.linspace(0.1, 1.0, 6),
            cv=cv_lc, scoring='r2', n_jobs=-1
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_sizes, train_sc.mean(axis=1),
                'o-', color='steelblue', label='Train', linewidth=2)
        ax.plot(train_sizes, val_sc.mean(axis=1),
                'o--', color='coral', label='Validare', linewidth=2)
        ax.fill_between(train_sizes,
                        train_sc.mean(axis=1) - train_sc.std(axis=1),
                        train_sc.mean(axis=1) + train_sc.std(axis=1),
                        alpha=0.15, color='steelblue')
        ax.fill_between(train_sizes,
                        val_sc.mean(axis=1) - val_sc.std(axis=1),
                        val_sc.mean(axis=1) + val_sc.std(axis=1),
                        alpha=0.15, color='coral')
        ax.set_xlabel('Număr exemple antrenare')
        ax.set_ylabel('R² Score')
        ax.set_title(f'Curbe de Învățare — {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # ── Predicție interactivă ─────────────────────────
    st.subheader('🎯 Predicție Interactivă')
    st.markdown('Introdu valorile pentru o mașină și obține prețul estimat:')

    c1, c2, c3 = st.columns(3)
    with c1:
        on_road_old = st.number_input('on road old (INR)',
                      min_value=500000, max_value=700000, value=600000, step=1000)
        on_road_now = st.number_input('on road now (INR)',
                      min_value=600000, max_value=900000, value=750000, step=1000)
        years       = st.slider('Vechime (ani)', 1, 10, 3)
        km          = st.number_input('Kilometraj', 10000, 200000, 80000, step=5000)

    with c2:
        rating      = st.slider('Rating (1-5)', 1, 5, 3)
        condition   = st.slider('Condiție (1-10)', 1, 10, 7)
        economy     = st.slider('Consum (km/l)', 8, 20, 14)

    with c3:
        top_speed   = st.slider('Viteză maximă (km/h)', 140, 210, 170)
        hp          = st.slider('Cai putere', 50, 110, 75)
        torque      = st.slider('Cuplu (Nm)', 68, 140, 100)

    if st.button('🔮 Prezice Prețul', use_container_width=True):
        input_data = pd.DataFrame([[
            on_road_old, on_road_now, years, km,
            rating, condition, economy, top_speed, hp, torque
        ]], columns=feature_names)

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]

        st.success(f'💰 Prețul estimat: **{prediction:,.0f} INR**')
        st.info(f'Model folosit: **{model_name}** · R² = {r2:.4f}')

        # Salvează pentru SHAP
        st.session_state['last_input']   = input_scaled
        st.session_state['last_input_df'] = input_data
        st.session_state['last_model']   = model_name

# ════════════════════════════════════════════════════
# TAB 3 — SHAP
# ════════════════════════════════════════════════════
with tab3:
    st.header('🔍 Explicabilitate SHAP')

    if 'last_input' not in st.session_state:
        st.info('💡 Fă o predicție în tab-ul "Modele & Predicție" '
                'pentru a vedea explicația SHAP.')
    else:
        model_used = st.session_state['last_model']
        model_shap = models[model_used]
        input_sc   = st.session_state['last_input']
        input_df   = st.session_state['last_input_df']

        st.markdown(f'**Explicație SHAP pentru modelul: {model_used}**')

        with st.spinner('Se calculează valorile SHAP...'):
            try:
                explainer   = shap.TreeExplainer(model_shap)
                shap_vals   = explainer.shap_values(input_sc)

                # ── Bar plot global pe test ────────────────
                st.subheader('Importanța globală a features')
                X_test_sc_df = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=feature_names
                )
                shap_vals_test = explainer.shap_values(
                    X_test_sc_df.sample(100, random_state=42)
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(
                    shap_vals_test,
                    X_test_sc_df.sample(100, random_state=42),
                    plot_type='bar',
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.close()

                # ── Force plot pentru predicția curentă ───
                st.subheader('Explicație locală — predicția ta')
                fig, ax = plt.subplots(figsize=(12, 3))
                shap.force_plot(
                    explainer.expected_value,
                    shap_vals[0],
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.close()

                # ── Waterfall ──────────────────────────────
                st.subheader('Contribuția fiecărei features')
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals[0],
                        base_values=explainer.expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.close()

            except Exception:
                # Fallback pentru modele non-tree (Linear, EBM)
                explainer = shap.Explainer(model_shap, input_sc)
                shap_exp  = explainer(input_sc)

                st.subheader('Contribuția fiecărei features')
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_exp[0], show=False)
                st.pyplot(plt.gcf())
                plt.close()