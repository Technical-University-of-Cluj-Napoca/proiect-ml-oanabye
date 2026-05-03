import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import learning_curve, KFold

st.set_page_config(page_title='Clasificare — Burnout', page_icon='🎓', layout='wide')

# ── Încărcare modele ──────────────────────────────────
@st.cache_resource
def load_models():
    models   = joblib.load('models/clf_models.pkl')
    scaler   = joblib.load('models/clf_scaler.pkl')
    features = joblib.load('models/clf_feature_names.pkl')
    return models, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv('data/student_burnout.csv')  # numele fișierului tău

models, scaler, feature_names = load_models()
df = load_data()

# ── Header ────────────────────────────────────────────
st.title('🎓 Predicția Burnout-ului Studențesc')
st.markdown('''
> *"E ora 2 noaptea, ai 3 examene săptămâna viitoare și te întrebi dacă
> munca pământului era o alegere mai bună. Felicitări — tocmai ai experimentat burnout academic."*
''')
st.markdown('---')

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
    col1.metric('Număr studenți', f'{len(df):,}')
    col2.metric('Features', '19')
    col3.metric('Clase', '3')
    col4.metric('Valori lipsă', '0')

    st.markdown('---')

    # Distribuția claselor
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Distribuția burnout_level')
        fig, ax = plt.subplots(figsize=(6, 4))
        df['burnout_level'].value_counts().plot(
            kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'],
            ax=ax, edgecolor='white'
        )
        ax.set_xlabel('Nivel Burnout')
        ax.set_ylabel('Număr studenți')
        ax.set_title('Distribuția claselor')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader('Distribuția anxiety_score')
        fig, ax = plt.subplots(figsize=(6, 4))
        df['anxiety_score'].hist(bins=20, color='steelblue',
                                  edgecolor='white', ax=ax)
        ax.set_xlabel('Anxiety Score')
        ax.set_ylabel('Frecvență')
        st.pyplot(fig)
        plt.close()

    # Boxplot burnout vs features cheie
    st.subheader('Features Cheie vs Burnout Level')
    key_features = ['anxiety_score', 'depression_score',
                    'academic_pressure_score', 'social_support_score',
                    'daily_sleep_hours', 'cgpa']
    order   = ['Low', 'Medium', 'High']
    palette = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_features):
        sns.boxplot(x='burnout_level', y=col, data=df,
                    order=order, palette=palette, ax=axes[i])
        axes[i].set_title(f'{col} vs burnout_level')
        axes[i].set_xlabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Heatmap corelații
    st.subheader('Matricea de Corelații')
    num_cols = ['age', 'daily_study_hours', 'daily_sleep_hours',
                'screen_time_hours', 'anxiety_score', 'depression_score',
                'academic_pressure_score', 'financial_stress_score',
                'social_support_score', 'physical_activity_hours',
                'attendance_percentage', 'cgpa']
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════
# TAB 2 — MODELE & PREDICȚIE
# ════════════════════════════════════════════════════
with tab2:
    st.header('Modele & Predicție')

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
        params    = model.get_params()
        params_df = pd.DataFrame(
            list(params.items()),
            columns=['Parametru', 'Valoare']
        )
        st.dataframe(params_df, use_container_width=True)

    # ── Metrici ───────────────────────────────────────
    with col2:
        st.subheader('📈 Metrici pe setul de test')

        # Reconstruiește X_test, y_test
        df_ml = df.copy()
        # Aplică același encoding ca în notebook
        ordinal_mappings = {
            'stress_level':    {'Low': 0, 'Medium': 1, 'High': 2},
            'sleep_quality':   {'Poor': 0, 'Average': 1, 'Good': 2},
            'internet_quality':{'Poor': 0, 'Average': 1, 'Good': 2},
            'year':            {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}
        }
        for col, mapping in ordinal_mappings.items():
            if col in df_ml.columns:
                df_ml[col] = df_ml[col].map(mapping)

        df_ml = pd.get_dummies(df_ml, columns=['gender', 'course'],
                               drop_first=True)
        target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df_ml['burnout_level'] = df_ml['burnout_level'].map(target_mapping)
        df_ml = df_ml.drop(columns=['student_id'], errors='ignore')

        X = df_ml.drop(columns=['burnout_level'])
        y = df_ml['burnout_level']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        X_test_sc = scaler.transform(X_test)
        y_pred    = model.predict(X_test_sc)
        y_proba   = model.predict_proba(X_test_sc)

        m1, m2 = st.columns(2)
        m1.metric('Accuracy', f'{accuracy_score(y_test, y_pred):.4f}')
        m2.metric('F1 Score', f'{f1_score(y_test, y_pred, average="weighted"):.4f}')
        m3, m4 = st.columns(2)
        m3.metric('Precision', f'{precision_score(y_test, y_pred, average="weighted"):.4f}')
        m4.metric('ROC-AUC',   f'{roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"):.4f}')

    st.markdown('---')

    # ── Matrice de confuzie ───────────────────────────
    st.subheader('Matricea de Confuzie')
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'], ax=ax)
    ax.set_xlabel('Prezis')
    ax.set_ylabel('Real')
    ax.set_title(f'Confusion Matrix — {model_name}')
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # ── Curbe de învățare ─────────────────────────────
    st.subheader('📉 Curbe de Învățare')
    with st.spinner('Se calculează...'):
        X_train_sc = scaler.transform(X_train)
        cv_lc = KFold(n_splits=3, shuffle=True, random_state=42)

        # Subset pentru viteză
        from sklearn.utils import resample
        X_lc, y_lc = resample(
            X_train_sc, y_train,
            n_samples=10000, random_state=42
        )

        train_sizes, train_sc, val_sc = learning_curve(
            model, X_lc, y_lc,
            train_sizes=np.linspace(0.1, 1.0, 6),
            cv=cv_lc, scoring='f1_weighted', n_jobs=-1
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
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Curbe de Învățare — {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown('---')

    # ── Predicție interactivă ─────────────────────────
    st.subheader('🎯 Predicție Interactivă')
    st.markdown('Completează profilul studentului:')

    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.slider('Vârstă', 18, 30, 21)
        study_hours   = st.slider('Ore studiu/zi', 0, 12, 4)
        sleep_hours   = st.slider('Ore somn/zi', 3, 10, 6)
        screen_time   = st.slider('Screen time (ore)', 0, 12, 4)

    with c2:
        anxiety       = st.slider('Anxiety score', 1, 10, 5)
        depression    = st.slider('Depression score', 1, 10, 5)
        stress        = st.selectbox('Stress level', [0, 1, 2],
                                      format_func=lambda x: ['Low','Medium','High'][x])
        academic_p    = st.slider('Academic pressure', 1, 10, 5)

    with c3:
        financial_s   = st.slider('Financial stress', 1, 10, 5)
        social_s      = st.slider('Social support', 1, 10, 5)
        attendance    = st.slider('Attendance %', 50, 100, 80)
        cgpa          = st.slider('CGPA', 1.0, 4.0, 3.0, step=0.1)
        physical      = st.slider('Physical activity (ore)', 0, 5, 1)

    if st.button('🔮 Prezice Burnout Level', use_container_width=True):
        input_data = pd.DataFrame([[
            age, study_hours, sleep_hours, screen_time,
            stress, anxiety, depression, academic_p,
            financial_s, social_s, physical, attendance, cgpa,
            1, 1, 0, 0, 0, 0, 0  # valori default pentru encoded columns
        ]], columns=feature_names)

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        pred_proba   = model.predict_proba(input_scaled)[0]

        label_map = {0: '🟢 Low', 1: '🟠 Medium', 2: '🔴 High'}
        color_map = {0: 'success', 1: 'warning', 2: 'error'}

        st.markdown(f'### Burnout Level Prezis: **{label_map[prediction]}**')

        prob_df = pd.DataFrame({
            'Clasă': ['Low', 'Medium', 'High'],
            'Probabilitate': pred_proba
        })
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.barh(prob_df['Clasă'], prob_df['Probabilitate'],
                color=colors, edgecolor='white')
        ax.set_xlabel('Probabilitate')
        ax.set_title('Distribuția probabilităților')
        st.pyplot(fig)
        plt.close()

        st.session_state['last_input']    = input_scaled
        st.session_state['last_input_df'] = input_data
        st.session_state['last_model']    = model_name

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
                explainer = shap.TreeExplainer(model_shap)
                shap_vals = explainer.shap_values(input_sc)

                # Bar plot global
                st.subheader('Importanța globală a features')
                X_test_sample = pd.DataFrame(
                    X_test_sc[:100], columns=feature_names
                )
                shap_test = explainer.shap_values(X_test_sample)

                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_test, X_test_sample,
                                  plot_type='bar',
                                  class_names=['Low', 'Medium', 'High'],
                                  show=False)
                st.pyplot(plt.gcf())
                plt.close()

                # Force plot
                st.subheader('Explicație locală — predicția ta')
                shap.force_plot(
                    explainer.expected_value[prediction],
                    shap_vals[prediction][0],
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.close()

            except Exception as e:
                st.warning(f'SHAP nu e disponibil pentru {model_used}: {e}')