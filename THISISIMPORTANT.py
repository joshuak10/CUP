from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

FPATH = r"C:\\Users\\awsom\\Downloads\\realdata.csv"

def _load_data():
    df = pd.read_csv(r"C:\\Users\\awsom\\Downloads\\realdata.csv")

    categorical_columns = [
 #       'icgc_donor_id',
 #       'project_code',
 #       'study_donor_involved_in',
  #      'submitted_donor_id',
   #     'donor_relapse_type',
    #    'donor_diagnosis_icd10',
     #   'donor_tumour_staging_system_at_diagnosis',
      #  'donor_tumour_stage_at_diagnosis',
      #  'donor_tumour_stage_at_diagnosis',
      #  'prior_malignancy',
      
    ]

    train_df = df.drop(columns=[
        'cancer_type', 'icgc_donor_id','project_code','study_donor_involved_in','submitted_donor_id',
         'donor_relapse_type','donor_diagnosis_icd10', 
         'donor_tumour_stage_at_diagnosis_supplemental','cancer_type_prior_malignancy', 'Unnamed: 0', 'Unnamed: 0.1',
         'donor_tumour_staging_system_at_diagnosis', 'disease_status_last_followup_no evidence of disease', 
         'donor_tumour_stage_at_diagnosis', 'prior_malignancy', 'disease_status_last_followup_relapse', 
         'donor_vital_status_deceased', 'cancer_history_first_degree_relative', 
         'disease_status_last_followup_no evidence of disease', 'donor_survival_time', 
         'disease_status_last_followup_progression','disease_status_last_followup_stable'
         
         ]) 

    train_df = pd.get_dummies(train_df, columns=categorical_columns)

    imputer = SimpleImputer(strategy='mean')
    feat_columns_encoded = imputer.fit_transform(train_df)
    
    pred_outcome = df['cancer_type']
    
    #feature_selector = SelectKBest(score_func=mutual_info_classif, k=30000)
    #feat_columns_selected = feature_selector.fit_transform(feat_columns_encoded, pred_outcome)
    k_best = SelectKBest(f_classif, k=(12000)) #Adjust 'k' as needed
    feat_columns_selected = k_best.fit_transform(feat_columns_encoded, pred_outcome)

    
    
    return train_df, feat_columns_selected, pred_outcome



def investigate_coeff(lr_model, train_df):
    import numpy as np

    # Assuming you already have the coefficients obtained from lr_model
    coefficients = lr_model.coef_  # Assuming it's a multiclass classification
    
    # Initialize lists to store top 10 feature names and coefficients for each class
    top_10_feature_names_per_class = []
    top_10_coefficients_per_class = []
    
    # Loop through each class
    for class_index in range(coefficients.shape[0]):
        class_coefficients = coefficients[class_index]
        class_names = lr_model.classes_
        # Get the absolute values of coefficients for sorting
        absolute_coefficients = np.abs(class_coefficients)
        
        # Get the indices that would sort the coefficients in descending order
        sorted_indices = np.argsort(absolute_coefficients)[::-1]
        
        # Get the top 10 indices and their corresponding feature names
        top_10_indices = sorted_indices[:10]
        top_10_feature_names = train_df.columns[top_10_indices].tolist()
        top_10_coefficients = class_coefficients[top_10_indices]
        
        top_10_feature_names_per_class.append(top_10_feature_names)
        top_10_coefficients_per_class.append(top_10_coefficients)
    
    # Print the top 10 feature names and coefficients for each class
    for class_index, (feature_names, coefficients) in enumerate(zip(top_10_feature_names_per_class, top_10_coefficients_per_class)):
        class_name = class_names[class_index]
        print(f"Class: {class_name}")
        for feature_name, coefficient in zip(feature_names, coefficients):
            print(f"  Feature: {feature_name}, Coefficient: {coefficient}")




def run():
    logger.info("Step1. Load data")
    train_df, feat_columns_selected, pred_outcome = _load_data()  # Updated variable name

    logger.info("Step2. Split")
    X_train, X_test, y_train, y_test = \
        train_test_split(feat_columns_selected, pred_outcome, test_size=0.2, random_state=42)  # Updated variable names

    print(X_train.shape)
    print(X_test.shape)
   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("Step3. Train Model")
    lr_model = LogisticRegression(penalty='l2', C=1)
    lr_model.fit(X_train, y_train)  # Use X_train here
    
    print(lr_model.coef_)
    investigate_coeff(lr_model, train_df)
    
    logger.info("Step4. Evaluate")
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    run()
