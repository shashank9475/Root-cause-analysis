import pandas as pd
import os
import glob
#from cluster_similarity_copy import individual_similarity
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kmean_v7 import cluster_solution
from Yake_v7 import Yake_main
import datatest as dt
from data_preprocessing_v7 import read_csv, data_cleaninig
from cluster_similarity_v7 import cluster_similarity_main
import pathlib
from sentence_transformers import SentenceTransformer
  

#UPLOAD_FOLDER = 'E:/Thesis/codes/final_codes/code_v6/UI/Upload_manager'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('bert-base-nli-mean-tokens')

@app.route('/')
def home():
    return render_template('home_page.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def empty_folder(path):
    files = list(pathlib.Path(path).glob('*.xlsx'))
    for f in files:
        os.remove(f)

@app.route('/manager_cluster', methods = ['GET','POST'])
def manager_cluster():
    if request.method == 'POST':
        count = [i for i in range(1,200)]
        
        
        if request.form['submit_button'] == 'Predict':
            file = request.files["file"]
            # if user does not select any file, browser also submit an empty part without filename
            if file.filename == '':
                return render_template('manager_cluster.html', message1="No file selected for upload")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                empty_folder("Upload_manager_single/excel")
                file.save(os.path.join("Upload_manager_single/excel", file.filename))
                
            excel_path = "Upload_manager_single/excel"
            pickle_path = "Upload_manager_single/pickle"
            if(validate_file(excel_path) == True):
                df_uploaded_cleaned = data_cleaninig(excel_path, pickle_path)
                df_uploaded_withoutClean = pd.read_pickle('Upload_manager_single/pickle/dataset_without_cleaning.pkl')
                
                df_main = pd.read_pickle('dataset/pickle/dataset.pkl')
                df_main_withoutClean = pd.read_pickle('dataset/pickle/dataset_without_cleaning.pkl')
                df_new = pd.concat([df_main, df_uploaded_cleaned])
                df_new_withoutClean = pd.concat([df_main_withoutClean, df_uploaded_withoutClean])
                
                cluster_list, df, df_withoutClean, list_yake = cluster_solution(df_new, df_new_withoutClean)
                
                
                message1 = "Here are the root cause of failure of all the failure groups"
                tables1=[i.to_html(classes='data', index=False) for i in cluster_list]
                zipped1 = zip(list_yake, tables1, count)
                return render_template('manager_cluster.html', zipped1=zipped1, titles1 =(i.columns.values for i in cluster_list), zip=zip, message1 = message1)
    
            else:
                message1 = "Wrong column: Please check column names or their sequence in the uploaded file."
                return render_template('manager_cluster.html', message1 = message1)
            
        elif request.form['submit_button'] == 'Add':
            file = request.files["file"]
            # if user does not select any file, browser also submit an empty part without filename
            if file.filename == '':
                return render_template('manager_cluster.html', message1="No file selected for upload")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                empty_folder("dataset/excel")
                file.save(os.path.join("dataset/excel", file.filename))
                
            excel_path = "dataset/excel"
            pickle_path = "dataset/pickle"
            if(validate_file(excel_path) == True):
                df_uploaded_cleaned = data_cleaninig(excel_path, pickle_path)
                df_uploaded_withoutClean = pd.read_pickle('dataset/pickle/dataset_without_cleaning.pkl')
                
                cluster_list, df, df_withoutClean, list_yake = cluster_solution(df_uploaded_cleaned, df_uploaded_withoutClean)
                
                message1 = "Here are the root cause of failure of all the failure groups"
                tables1=[i.to_html(classes='data', index=False) for i in cluster_list]
                zipped1 = zip(list_yake, tables1, count)
                return render_template('manager_cluster.html', zipped1=zipped1, titles1 =(i.columns.values for i in cluster_list), zip=zip, message1 = message1)
    
            else:
                message1 = "Wrong column: Please check column names or their sequence in the uploaded file."
                return render_template('manager_cluster.html', message1 = message1)
            
        elif request.form['submit_button'] == 'See full cluster':
            df = pd.read_pickle('dataset/pickle/dataset.pkl')
            df_withoutClean = pd.read_pickle('dataset/pickle/dataset_without_cleaning.pkl')
                
            cluster_list, df, df_withoutClean, list_yake = cluster_solution(df, df_withoutClean)
                
            message1 = "Here are the root causes of all the failure groups"
            tables1=[i.to_html(classes='data', index=False) for i in cluster_list]
            zipped1 = zip(list_yake, tables1, count)
            return render_template('manager_cluster.html', zipped1=zipped1, titles1 =(i.columns.values for i in cluster_list), zip=zip, message1 = message1)
        
    return render_template('manager_cluster.html', message="Success")



@app.route('/upload_manager', methods = ['GET','POST'])
def upload_manager():
    if request.method == 'POST':
        text = request.form.get('text')  
        count = [i for i in range(1,200)]
        if text:
            print("text file method")
            list_cluster, list_yake, individual_datasample, list_yake_indi = cluster_similarity_main(text, model)

            if not list_cluster:
                message1 = "No similar dataset"
                return render_template('upload_manager.html', message1 = message1)
            else:
                if not individual_datasample:
                    message1 = "Here are the root causes of failure for clusters having similar failure"
                    tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                    zipped1 = zip(list_yake, tables1, count)
                    return render_template('upload_manager.html', zipped1=zipped1, titles1 =(i.columns.values for i in list_cluster), zip=zip, message1 = message1)
                
                else:
                    message1 = "Here are the root causes of failure for clusters having similar failure"
                    message2 = "Here are the root causes of failure for individual failure data which are not there in above clusters"
                    tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                    tables2=[i.to_html(classes='data', index=False) for i in individual_datasample]
                    zipped1 = zip(list_yake, tables1, count)
                    zipped2 = zip(list_yake_indi, tables2, count)
                    return render_template('upload_manager.html', zipped1=zipped1, zipped2=zipped2, titles1 =(i.columns.values for i in list_cluster), titles2 =(i.columns.values for i in individual_datasample), zip=zip, message1 = message1, message2 = message2)



        else:
            print("upload file method")
            
            file = request.files["file"]
            # if user does not select any file, browser also submit an empty part without filename
            if file.filename == '':
                return render_template('upload_manager.html', message1="No file selected for upload")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                empty_folder("Upload")
                file.save(os.path.join("Upload", file.filename))
            path = "Upload"
            if(validate_file(path) == True):
                text3 = cluster_list_display()
                print(text3)
                if text3:
                    list_cluster, list_yake, individual_datasample, list_yake_indi = cluster_similarity_main(text3, model)
                    
                    if not list_cluster:
                        message1 = "No similar document"
                        return render_template('upload_manager.html', message1 = message1)
                    else:
                        if not individual_datasample:
                            message1 = "Here are the root causes of failure for clusters having similar failure"
                            tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                            zipped1 = zip(list_yake, tables1, count)
                            return render_template('upload_manager.html', zipped1=zipped1, titles1 =(i.columns.values for i in list_cluster), zip=zip, message1 = message1)
                
                        else:
                            message1 = "Here are the root causes of failure for clusters having similar failure"
                            message2 = "Here are the root causes of failure for individual failure data which are not there in above clusters"
                            tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                            tables2=[i.to_html(classes='data', index=False) for i in individual_datasample]
                            zipped1 = zip(list_yake, tables1, count)
                            zipped2 = zip(list_yake_indi, tables2, count)
                            return render_template('upload_manager.html', zipped1=zipped1, zipped2=zipped2, titles1 =(i.columns.values for i in list_cluster), titles2 =(i.columns.values for i in individual_datasample), zip=zip, message1 = message1, message2 = message2)
                
                else:
                    message1 = "Type of failure column or Technical column is empty. Please make sure atleast any one of these column has data."
                    return render_template('upload_manager.html', message1 = message1)
            else:
                message1 = "Wrong column: Please check column names or their sequence in the uploaded file."
                return render_template('upload_manager.html', message1 = message1)
    
    return render_template('upload_manager.html', message1="Success")   


def validate_file(path):
    list_of_files = list(pathlib.Path(path).glob('*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    
    df = read_csv(latest_file)
    column_list = ["Target version","Date","Circ status","Block","Index","Type of failure","Technical explanation","Modification concept","To be measured or simulated","Schematic modification","Layout modification","Modification of test program","Product specification modification","Measures against recurrence","Analog","Digital","Test","Layout","Software","Quality","Spec","Additional Information","Others"]
    if(list(df.columns) == column_list):
        return True
    else:
        return False
    
def cluster_list_display():
    list_of_files = glob.glob('Upload/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    
    df2 = read_csv(latest_file)
    text1 = ""
    text2 = ""
    if(df2["Technical explanation"].dropna().empty == False):
        text1 = df2["Technical explanation"].iloc[0]
    if(df2["Type of failure"].dropna().empty == False):
        text2 = df2["Type of failure"].iloc[0]
    text = text1 + " " + text2           
    return text

@app.route('/upload_user', methods = ['GET','POST'])
def upload_user():
    if request.method == 'POST':
        text = request.form.get('text')  
        count = [i for i in range(1,200)]
        if text:
            print("text file method")
            list_cluster, list_yake, individual_datasample, list_yake_indi = cluster_similarity_main(text, model)

            if not list_cluster:
                message1 = "No similar dataset"
                return render_template('upload_user.html', message1 = message1)
            else:
                if not individual_datasample:
                    message1 = "Here are the root causes of failure for clusters having similar failure"
                    tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                    zipped1 = zip(list_yake, tables1, count)
                    return render_template('upload_user.html', zipped1=zipped1, titles1 =(i.columns.values for i in list_cluster), zip=zip, message1 = message1)
                
                else:
                    message1 = "Here are the root causes of failure for clusters having similar failure"
                    message2 = "Here are the root causes of failure for individual failure data which are not there in above clusters"
                    tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                    tables2=[i.to_html(classes='data', index=False) for i in individual_datasample]
                    zipped1 = zip(list_yake, tables1, count)
                    zipped2 = zip(list_yake_indi, tables2, count)
                    return render_template('upload_user.html', zipped1=zipped1, zipped2=zipped2, titles1 =(i.columns.values for i in list_cluster), titles2 =(i.columns.values for i in individual_datasample), zip=zip, message1 = message1, message2 = message2)



        else:
            print("upload file method")
            
            file = request.files["file"]
            # if user does not select any file, browser also submit an empty part without filename
            if file.filename == '':
                return render_template('upload_user.html', message1="No file selected for upload")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                empty_folder("Upload")
                file.save(os.path.join("Upload", file.filename))
                #return render_template('upload_user.html', message="Success")
            path = "Upload"
            if(validate_file(path) == True):
                text3 = cluster_list_display()
                print(text3)
                if text3:
                    list_cluster, list_yake, individual_datasample, list_yake_indi = cluster_similarity_main(text3, model)
                    
                    if not list_cluster:
                        message1 = "No similar document"
                        return render_template('upload_user.html', message1 = message1)
                    else:
                        if not individual_datasample:
                            message1 = "Here are the root causes of cluster having similar failures"
                            tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                            zipped1 = zip(list_yake, tables1, count)
                            return render_template('upload_user.html', zipped1=zipped1, titles1 =(i.columns.values for i in list_cluster), zip=zip, message1 = message1)
                
                        else:
                            message1 = "Here are the root causes of cluster having similar failures"
                            message2 = "Some other Root causes for similar failures other than the above clusters"
                            tables1=[i.to_html(classes='data', index=False) for i in list_cluster]
                            tables2=[i.to_html(classes='data', index=False) for i in individual_datasample]
                            zipped1 = zip(list_yake, tables1, count)
                            zipped2 = zip(list_yake_indi, tables2, count)
                            return render_template('upload_user.html', zipped1=zipped1, zipped2=zipped2, titles1 =(i.columns.values for i in list_cluster), titles2 =(i.columns.values for i in individual_datasample), zip=zip, message1 = message1, message2 = message2)
                
                else:
                    message1 = "Type of failure column or Technical column is empty. Please make sure atleast any one of these column has data."
                    return render_template('upload_user.html', message1 = message1)
            else:
                message1 = "Wrong column: Please check column names or their sequence in the uploaded file."
                return render_template('upload_user.html', message1 = message1)
    
    return render_template('upload_user.html', message1="Success")        
        




if __name__ == '__main__':
    app.run(debug=False)
    #app.run(host='localhost', port=5000)