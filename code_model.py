import streamlit as st
import numpy as np
import xgboost as xgb


xgb_classifier = xgb.XGBClassifier()
xgb_classifier.load_model("xgboost_model.json")

def preprocess_data(input_data):
    # Tách chuỗi dữ liệu và gán các giá trị vào biến
    age = input_data[0]  # age
    job = input_data[1]  # job
    marital = input_data[2]  # marital status
    education = input_data[3]  # education
    default = input_data[4]  # default
    balance = input_data[5]  # balance
    housing = input_data[6]  # housing
    loan = input_data[7]  # loan
    contact = input_data[8]  # contact
    day = input_data[9]  # day
    month = input_data[10]  # month
    duration = input_data[11]  # duration
    campaign = input_data[12]  # campaign
    pdays = input_data[13]  # pdays
    previous = input_data[14]  # previous
    poutcome = input_data[15]  # poutcome

    age_min, age_max = 18, 72
    balance_min, balance_max = -1884, 3415
    duration_min, duration_max = 0, 3881
    day_min, day_max = 1, 31
    age_normalized = round((age - age_min) / (age_max - age_min), 3)
    balance_normalized = round((balance - balance_min) / (balance_max - balance_min), 3)
    duration_normalized = round((duration - duration_min) / (duration_max - duration_min), 3)
    day_normalized = round((day - day_min) / (day_max - day_min), 3)

    # Mã hóa nhị phân cho các biến
    job_encoding = {
        "management": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "retired": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "unemployed": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "student": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "technician": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "entrepreneur": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "admin": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "services": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "blue_collar": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "self_employed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "housemaid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    marital_encoding = {
        "married": [1, 0, 0],
        "single": [0, 1, 0],
        "divorced": [0, 0, 1],
    }

    education_encoding = {
        "tertiary": [1, 0, 0],
        "primary": [0, 1, 0],
        "secondary": [0, 0, 1],
    }

    default_encoding = 1 if default == "yes" else 0
    housing_encoding = 1 if housing == "yes" else 0
    loan_encoding = 1 if loan == "yes" else 0
    
    contact_encoding = {
        "unknown": [1, 0, 0],
        "cellular": [0, 1, 0],
        "telephone": [0, 0, 1],
    }

    campaign_encoding = {
        1: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # campaign 1
        2: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # campaign 2
        3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # campaign 3
        4: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # campaign 4
        5: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # campaign 5
        6: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # campaign 6
        7: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # campaign 7
        8: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # campaign 8
        9: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # campaign 9
        10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # campaign 10
        11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # campaign 11
    }

    month_encoding = {
        "may": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "jun": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "jul": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "aug": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "oct": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "dec": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "feb": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "mar": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "apr": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "sep": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "nov": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "jan": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    poutcome_encoding = {
        "unknown": [1, 0, 0],
        "success": [0, 1, 0],
        "failure": [0, 0, 1],
    }

    
    features = np.array([
        age_normalized,
        *job_encoding.get(job, [0]*11), 
        *marital_encoding.get(marital, [0]*3),  
        *education_encoding.get(education, [0]*3), 
        balance_normalized,
        housing_encoding,
        loan_encoding,
        *contact_encoding.get(contact, [0]*3),  
        day_normalized,
        *month_encoding.get(month, [0]*12),  
        duration_normalized,
        *campaign_encoding.get(campaign, [0]*11),  
        pdays,
        previous,
        *poutcome_encoding.get(poutcome, [0]*3),  
    ])
    
    return features
# Tạo giao diện nhập dữ liệu với Streamlit
st.title("Dự đoán mô hình")

# Tạo 3 cột
col1, col2, col3 = st.columns(3)

# Cột 1
with col1:
    age = st.number_input("Tuổi", min_value=18, max_value=72, value=30)
    job = st.selectbox("Nghề nghiệp", ["admin", "management", "retired", "unemployed", "student", "technician", "entrepreneur", "services", "blue_collar", "self_employed", "housemaid"])
    marital = st.selectbox("Tình trạng hôn nhân", ["married", "single", "divorced"])

# Cột 2
with col2:
    education = st.selectbox("Trình độ học vấn", ["tertiary", "primary", "secondary"])
    default = st.selectbox("Nợ xấu", ["yes", "no"])
    balance = st.number_input("Số dư", min_value=-1884, max_value=3415, value=0)

# Cột 3
with col3:
    housing = st.selectbox("Vay mua nhà", ["yes", "no"])
    loan = st.selectbox("Vay cá nhân", ["yes", "no"])
    contact = st.selectbox("Phương thức liên lạc", ["unknown", "cellular", "telephone"])

# Dòng thứ hai của lưới với 3 cột
col4, col5, col6 = st.columns(3)

# Cột 4
with col4:
    day = st.number_input("Ngày gọi", min_value=1, max_value=31, value=1)
    month = st.selectbox("Tháng gọi", ["may", "jun", "jul", "aug", "oct", "dec", "feb", "mar", "apr", "sep", "nov", "jan"])

# Cột 5
with col5:
    duration = st.number_input("Thời gian liên hệ", min_value=0, max_value=3881, value=0)
    campaign = st.number_input("Số chiến dịch", min_value=1, max_value=11, value=1)

# Cột 6
with col6:
    pdays = st.number_input("Số ngày từ lần liên hệ trước", min_value=0, max_value=999, value=0)
    previous = st.number_input("Lần liên hệ trước", min_value=0, max_value=99, value=0)
    poutcome = st.selectbox("Kết quả liên hệ trước", ["unknown", "success", "failure"])


input_data = [age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]
features = preprocess_data(input_data)

if st.button("Dự đoán"):
    # Chuyển đổi đặc trưng đã chuẩn hóa thành numpy array
    normalized_vector = np.array(features)
    n_features = 55
    X_full = np.zeros(n_features)

    for i in range(len(normalized_vector)):
        X_full[i] = normalized_vector[i]

    X_full = X_full.reshape(1, -1)  # Chuyển thành 2D array với 1 hàng (1 sample)
    prediction = xgb_classifier.predict(X_full)

    st.write(f"Đặc trưng đã xử lý: {features}")
    st.write(f"Dự đoán: {prediction[0]}")
