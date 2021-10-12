from pycaret.classification import *
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle5 as pickle
import time
import shap
from PIL import Image
import joblib


if __name__ == '__main__':
    # Load the model
    model = load_model('deployment_28042020')
    recall_optimised = load_model('XGBoost_recall')
    pipeline = joblib.load('pipeline.pkl')



    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['Label'][0]

        score = predictions_df['Score'][0]
        return predictions


    def predict_GEN(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)

        return predictions_df

    def explain():
        explanation = Image.open('summary.png')
        explanation.thumbnail((500, 500))
        st.image(explanation)


    def probability(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        score = predictions_df['Score'][0]
        return score


    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    ######## Footer

    #########

    def run():
        #from PIL import Image

        image = Image.open('logo.jpg')
        image.thumbnail((150, 150))
        pycaret_image = Image.open('hosp.png')
        # AI = Image.open('ai3.jpg')
        # AI.thumbnail((500, 400))
        st.set_page_config(page_title='Botswana Life Lapse Engine', initial_sidebar_state="expanded")
        st.sidebar.markdown('**Powered by**')
        padding = 0
        st.markdown(f""" <style>
            .reportview-container .main .block-container{{
                padding-top: {padding}rem;
                padding-right: {padding}rem;
                padding-left: {padding}rem;
                padding-bottom: {padding}rem;
            }} </style> """, unsafe_allow_html=True)

        page_bg_img = '''
        <style>
        body {
        background-image: url("https://i.stack.imgur.com/2wV8x.png");
        background-size: cover;
        }
        </style>
        '''

        #####

        ####

        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.image(image)
        # st.image(AI)

        st.sidebar.image(pycaret_image)

        add_selector_box = st.sidebar.radio('Please select Prediction Type', ['Online', 'Batch'])

        st.title('Botswana Life Lapse Prediction Engine')
        # Online predictions
        add_selector_box_model2 = st.radio('Please select Prediction Model_Type', ['Default', 'Recall Optimised'])
        if add_selector_box == 'Online':

            my_expander = st.expander(label='Show Input Policy/Customer Variables')
            with my_expander:
                # 'Hello there!'
                # clicked = st.button('Click me!')
                st.sidebar.info(' Categorical Paramters')
                PRODUCT_CODE = st.selectbox('Product Code', ['BMFW-1', 'ULP1-1', 'ULM2-1', 'URA1-1', 'ULK6-1', 'ULK1-1',
                                                             'ULP3-1', 'BMKS-1', 'ULM1-1', 'ULM6-1', 'ULMG-1', 'ULM5-1',
                                                             'ULM3-1', 'BIF5-1', 'ULK2-1', 'BMPS-1', 'ULG1-1',
                                                             'IGI1-44',
                                                             'BIF3-1', 'SCEP-1', 'BMKJ-1', 'MWLP-1', 'UMAT-1',
                                                             'SCHP-13',
                                                             'SCHP-9', 'SCHP-1', 'LORE', 'SCFP-1', 'BMK1-1', 'SCTFP',
                                                             'BMPJ-1',
                                                             'SCTEP', 'PPLC', 'BPHC', 'MPFP', 'ULM7-1', 'URAG-1',
                                                             'SCHP-5',
                                                             'SPP1', 'SPRA-1', 'SPOI', 'PSTA-1', 'MFRA-1', 'MFDS-1',
                                                             'PSTA-2',
                                                             'BPSTA-1', 'PSTA-4', 'PSTA-4-2', 'PSTA-4-3', 'BPSTA-2',
                                                             'MFDS-2',
                                                             'PSWL-1', 'UBRA-1', 'SCLR-1', 'SCPS1', 'PSWL-2', 'SCPS2',
                                                             'BPHC2',
                                                             'SCHOS', 'MSKO-1', 'ELRE-1', 'MOSK-1', 'WPLA-1', 'MOSK60',
                                                             'TSWIP1', 'MOSK65'])

                # DOC=st.date_input('Date of Commencement')
                Policy_Age = st.number_input('Policy Base', min_value=0, max_value=29999999)
                # EXPIRY_DATE = st.date_input('When does the policy expire')
                PRODUCT_TYPE = st.selectbox('Product Class', ['Risk', 'Investment'])
                TERM = st.number_input('TERM', min_value=0, max_value=29999999)
                PREMIUM_AMOUNT = st.number_input('premium', min_value=0, max_value=29999999)
                SA = st.number_input('SA', min_value=0, max_value=29999999)
                PREMIUM_STATUS = st.sidebar.selectbox('Premium Status',
                                                      ['Regular', 'ILP Premium Paid-Up', 'Fully Paid',
                                                       'Premium Waived'])
                PREMIUM_FREQUENCY = st.selectbox('PREMIUM_FREQUENCY',
                                                 ['Monthly', 'Yearly', 'Half Yearly', 'Quarterly', 'Single'])
                MODE_NAME = st.sidebar.selectbox('PayMode',
                                                 ['DDE', 'Cash', 'GSO', 'ESO(Semi-Electronic)', 'ESO(Electronic)',
                                                  'Cheque',
                                                  'BSO', 'Electronic Cash Collection'])
                GENDER = st.sidebar.selectbox('GENDER', ['F', 'M'])
                MARITAL_STATUS = st.sidebar.selectbox('MARITAL_STATUS',
                                                      ['Single', 'Married', 'Divorced', 'Other', 'Widowed',
                                                       'Separated'])
                Policy_holder_Age = st.number_input('Age', min_value=0, max_value=100)
                INCOME = st.number_input('Income(Yearly)', min_value=0, max_value=100000000)
                SALES_CHANNEL = st.sidebar.selectbox('Channel', ['Broker', 'Tied'])
                AGENT_PERSISTENCY = st.number_input('Agent Persistency', min_value=0, max_value=100)
                EDUCATION = st.sidebar.selectbox('Education', ['Others', 'Unknown'])

            Prediction = ''
            input_dict = {'AGENT_PERSISTENCY': float(AGENT_PERSISTENCY), 'GENDER': GENDER, 'INCOME': float(INCOME),
                          'MARITAL_STATUS': MARITAL_STATUS, 'MODE_NAME': MODE_NAME, 'PREMIUM_AMOUNT': float(PREMIUM_AMOUNT),
                          'PREMIUM_FREQUENCY': PREMIUM_FREQUENCY, 'PREMIUM_STATUS': PREMIUM_STATUS,
                          'PRODUCT_CODE': PRODUCT_CODE, 'PRODUCT_TYPE': PRODUCT_TYPE, 'Policy_Age': float(Policy_Age),
                          'Policy_holder_Age': float(Policy_holder_Age), 'SA': float(SA),
                          'SALES_CHANNEL': SALES_CHANNEL, 'TERM': int(TERM), 'EDUCATION': EDUCATION}
            input_df = pd.DataFrame([input_dict])
            if add_selector_box_model2 == 'Recall Optimised':
                Prediction_class = predict(model=recall_optimised, input_df=input_df)
                probabi = probability(model=recall_optimised, input_df=input_df)
            else:
                Prediction_class = predict(model=model, input_df=input_df)
                probabi = probability(model=model, input_df=input_df)
            if st.button('Predict'):

                # Spinner

                # Progress Bar
                latest_iteration = st.empty()
                bar = st.progress(0)
                for i in range(100):
                    # Update the progress bar with each iteration.
                    latest_iteration.text(f'Computing...⏳ {i + 1}%')
                    bar.progress(i + 1)
                    time.sleep(0.1)

                st.warning("Computation complete ✔")
                # st.balloons()
                Prediction = 'Predicted class is ' + Prediction_class + ' with confidence ' + str(probabi * 100) + '%'+' using '+add_selector_box_model2

                st.success(Prediction)
            if st.button('Explain Predictions'):
                    def st_shap(plot, height=None):
                        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                        components.html(shap_html, height=height)

                    #import pickle
                    import matplotlib.pyplot as plt

                    with open('score_objects.pkl', 'rb') as handle:
                        d, features_selected, xgboost, explainer = pickle.load(handle)


                    def explain_model_prediction(data1):
                        # Calculate Shap values
                        shap_values = explainer.shap_values(data1)
                        #st.write(shap_values) #This was just to see the values
                        p = shap.force_plot(explainer.expected_value, shap_values, data1)
                        return p, shap_values

                    st.header('Final Result')
                    prediction = Prediction_class

                    probability_value = probabi

                    st.write("Prediction: ", prediction)
                    st.write("Probability: ", round(float(probability_value), 3))
                    input_df.to_csv('treyyyyy.csv')
                    input_df['STATUS_NAME'] = prediction
                    transformed=pipeline.transform(input_df)
                    st.subheader('Transformed input data')
                    st.write(transformed)
                    # explainer force_plot
                    #Prediction.drop(['Label', 'Score'], axis=1, inplace=True)
                    #results = predict_GEN(model, input_df)
                    #st.write(results)
                    #input_df.drop(['Policy_holder_Age'],axis=1,inplace=True)
                    #input_df.to_csv('treyyyyy.csv')
                    #explain()
                    #results.drop(['Label', 'Score'], axis=1, inplace=True)
                    p, shap_values = explain_model_prediction(transformed)
                    st.subheader('Model Prediction Interpretation Plot')
                    st_shap(p)

                    st.subheader('Summary Plot 1')
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    shap.summary_plot(shap_values, transformed)
                    st.pyplot(fig)

                    st.subheader('Summary Plot 2')
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    shap.summary_plot(shap_values, transformed, plot_type='bar')
                    st.pyplot(fig)

        if add_selector_box == 'Batch':

            file_upload = st.file_uploader('Please upload CSV file for predictions', type=['csv'])

            if file_upload is not None:
                data = pd.read_csv(file_upload)
                # Progress Bar
                latest_iteration = st.empty()
                bar = st.progress(0)
                for i in range(100):
                    # Update the progress bar with each iteration.
                    latest_iteration.text(f'Computing...⏳ {i + 1}%')
                    bar.progress(i + 1)
                    time.sleep(0.1)
                predictions = predict_model(estimator=model, data=data)
                st.sidebar.success('Predictions Complete ✔')
                st.write(predictions)
                ############################
                if st.button('Why'):
                    def st_shap(plot, height=None):
                        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                        components.html(shap_html, height=height)

                    import pickle
                    import matplotlib.pyplot as plt

                    with open('score_objects.pkl', 'rb') as handle:
                        d, features_selected, clf, explainer = pickle.load(handle)

                    st.write('Kuda')

                    def explain_model_prediction(data1):
                        # Calculate Shap values
                        shap_values = explainer.shap_values(data1)
                        p = shap.force_plot(explainer.expected_value[1], shap_values[1], data1)
                        return p, shap_values

                    st.header('Final Result')
                    prediction = predictions["Label"]

                    probability_value = predictions["Score"]

                    st.write("Prediction: ", prediction)
                    st.write("Probability: ", round(float(probability_value), 3))

                    # explainer force_plot
                    predictions.drop(['prediction', 'probability'], axis=1, inplace=True)
                    results = predictions[features_selected]
                    res=predict(pipeline,input_df)
                    st.write(res)
                    p, shap_values = explain_model_prediction(results)
                    st.subheader('Model Prediction Interpretation Plot')
                    st_shap(p)

                    st.subheader('Summary Plot 1')
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    shap.summary_plot(shap_values[1], results)
                    st.pyplot(fig)

                    st.subheader('Summary Plot 2')
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    shap.summary_plot(shap_values[1], results, plot_type='bar')
                    st.pyplot(fig)
                ############################

                # st.balloons()
                # Download the csv version of the preditions
                if st.checkbox('Analyse Predicted data in a new browser'):
                    import dtale as dt
                    dashboard = dt.show(predictions, ignore_duplicate=True)
                    dashboard.open_browser()
                    # st.subheader('Raw data')
                    # st.write(data)
                if st.checkbox('Compute Pair Plots'):
                    import seaborn as sns
                    st.title("Please wait while we render your visuals")
                    fig = sns.pairplot(predictions, hue="Label")
                    st.pyplot(fig)

                    # st.bar_chart(predictions['Label'])

                    # hist_values = np.histogram(predictions['Label'], bins=24, range=(0, 24))[0]

                csv = convert_df(predictions)
                st.download_button(
                    "Download Predictions",
                    csv,
                    "Predictions.csv",
                    "text/csv",
                    key='browser-data'
                )


    run()
