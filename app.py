from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # gunakan backend non-GUI agar aman di server/thread
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('credit_card_model.pkl')
scaler = joblib.load('credit_card_scaler.pkl')

# Feature columns yang digunakan untuk training
feature_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

@app.route('/')
def home():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk prediksi"""
    try:
        # Terima JSON (AJAX) atau form (HTML klasik)
        if request.is_json:
            data = request.get_json(silent=True) or {}
            wants_json = True
        else:
            data = request.form
            wants_json = False

        # Validasi input
        required_fields = [
            'limit_bal', 'sex', 'education', 'marriage', 'age',
            'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
            'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'
        ]

        missing = [f for f in required_fields if f not in data or str(data[f]).strip() == '']
        if missing:
            error_msg = f"Field berikut wajib diisi: {', '.join(missing)}"
            return (jsonify({'error': error_msg}), 400) if wants_json else render_template('result.html', hasil_prediksi={'error': error_msg})

        # Parsing dan tipe data dengan validasi yang jelas
        def to_int(name):
            try:
                return int(str(data[name]).strip())
            except Exception:
                raise ValueError(f"Nilai '{name}' harus berupa angka bulat.")

        def to_float(name):
            try:
                return float(str(data[name]).strip())
            except Exception:
                raise ValueError(f"Nilai '{name}' harus berupa angka.")

        input_data = [
            to_float('limit_bal'),
            to_int('sex'),
            to_int('education'),
            to_int('marriage'),
            to_int('age'),
            to_int('pay_0'),
            to_int('pay_2'),
            to_int('pay_3'),
            to_int('pay_4'),
            to_int('pay_5'),
            to_int('pay_6'),
            to_float('bill_amt1'),
            to_float('bill_amt2'),
            to_float('bill_amt3'),
            to_float('bill_amt4'),
            to_float('bill_amt5'),
            to_float('bill_amt6'),
            to_float('pay_amt1'),
            to_float('pay_amt2'),
            to_float('pay_amt3'),
            to_float('pay_amt4'),
            to_float('pay_amt5'),
            to_float('pay_amt6')
        ]

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        risk_level = "Berisiko Gagal Bayar" if prediction == 1 else "Tidak Berisiko"
        risk_probability = probability[1] * 100

        if risk_probability < 30:
            interpretation = "Risiko Rendah"
            color = "success"
        elif risk_probability < 60:
            interpretation = "Risiko Sedang"
            color = "warning"
        else:
            interpretation = "Risiko Tinggi"
            color = "danger"

        hasil_prediksi = {
            'prediction': int(prediction),
            'risk_level': risk_level,
            'risk_probability': round(risk_probability, 2),
            'interpretation': interpretation,
            'color': color,
            'non_default_prob': round(probability[0] * 100, 2)
        }

        # --- Simpan grafik ke static (seperti sebelumnya) ---
        from sklearn.metrics import confusion_matrix
        y_true = [hasil_prediksi['prediction']]
        y_pred = [hasil_prediksi['prediction']]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.hist([input_data[4]], bins=10, color='skyblue')
        plt.title('Age Distribution')
        plt.savefig('static/age_distribution.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.hist([input_data[0]], bins=10, color='orange')
        plt.title('Credit Limit Distribution')
        plt.savefig('static/limitbal_distribution.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.bar(['No Default', 'Default'], [hasil_prediksi['non_default_prob'], hasil_prediksi['risk_probability']])
        plt.title('Target Distribution')
        plt.savefig('static/target_distribution.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.bar(['LIMIT_BAL'], [input_data[0]])
        plt.title('LIMIT_BAL vs Default')
        plt.savefig('static/limitbal_vs_default.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.bar(['SEX'], [input_data[1]])
        plt.title('SEX vs Default')
        plt.savefig('static/sex_vs_default.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.bar(['EDUCATION'], [input_data[2]])
        plt.title('EDUCATION vs Default')
        plt.savefig('static/education_vs_default.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.bar(['MARRIAGE'], [input_data[3]])
        plt.title('MARRIAGE vs Default')
        plt.savefig('static/marriage_vs_default.png')
        plt.close()

        plt.figure(figsize=(4, 4))
        dummy_corr = np.eye(len(input_data))
        sns.heatmap(dummy_corr, annot=False, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('static/correlation_heatmap.png')
        plt.close()
        # --- Akhir simpan grafik ---

        return jsonify(hasil_prediksi) if wants_json else render_template('result.html', hasil_prediksi=hasil_prediksi)
        
    except Exception as e:
        # Log sederhana ke console untuk debugging lokal
        print('ERROR /predict:', repr(e))
        return (jsonify({'error': str(e)}), 500) if request.is_json else render_template('result.html', hasil_prediksi={'error': str(e)})

@app.route('/about')
def about():
    """Halaman tentang aplikasi"""
    return render_template('about.html')

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction dari CSV upload atau default dataset."""
    try:
        uploaded = request.files.get('file')
        use_default = request.form.get('use_default') == '1'

        if uploaded and uploaded.filename:
            df = pd.read_csv(uploaded)
        elif use_default or not uploaded:
            df = pd.read_csv('UCI_Credit_Card.csv')
        else:
            return jsonify({'error': 'Harap unggah file CSV atau pilih gunakan dataset bawaan.'}), 400

        # Pastikan semua kolom fitur tersedia
        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f"Kolom berikut tidak ditemukan di CSV: {', '.join(missing_cols)}"}), 400

        X = df[feature_columns].copy()

        # Coerce ke numerik dan tangani nilai tak valid
        for col in feature_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

        # Scale dan prediksi
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)

        result_df = df.copy()
        result_df['prediction'] = preds
        result_df['default_probability'] = (probas[:, 1] * 100).round(2)
        result_df['non_default_probability'] = (probas[:, 0] * 100).round(2)
        result_df['risk_level'] = np.where(preds == 1, 'Berisiko Gagal Bayar', 'Tidak Berisiko')
        result_df['risk_probability'] = result_df['default_probability']

        ts = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        out_name = f'predictions_{ts}.csv'
        out_path = os.path.join('static', out_name)
        result_df.to_csv(out_path, index=False)

        return jsonify({
            'message': 'Batch prediksi berhasil.',
            'download_url': url_for('static', filename=out_name)
        })
    except Exception as e:
        print('ERROR /predict_batch:', repr(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

