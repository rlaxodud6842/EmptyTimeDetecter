from flask import Flask, json, redirect, render_template, request, url_for
import os
import uuid
import ImageProccsser as ImageProccsser
import EmptyTimeDetecter as ED

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "❌ 파일이 안 왔어!"
    
    files = request.files.getlist('file') #여러개 파일 리스트 받기
    
    job_id = str(uuid.uuid4())  # 고유 ID 생성
    upload_dir = os.path.join("uploads", job_id)
    os.makedirs(upload_dir, exist_ok=True)
            
    for file in files:
        if file.filename == '':
            return "❌ 파일 이름이 비었어!"
        
        if file:
            filepath = os.path.join(upload_dir, file.filename)
            file.save(filepath)
    
    image_processer = ImageProccsser.imageProcessor()
    lecture_time = image_processer.process_folder(upload_dir)#수업인 시간을 반환함.

    meet = {job_id : lecture_time}
    empty_time = ED.filter_table(meet)
    
    result_path = os.path.join(upload_dir, "result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(empty_time, f)

    return redirect(url_for('show_result', job_id=job_id))

@app.route('/result/<job_id>')
def show_result(job_id):
    result_path = os.path.join("uploads", job_id, "result.json")
    if not os.path.exists(result_path):
        return "결과 없음", 404
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[0]
    
    return render_template("result.html", converted_schedule=data)

if __name__ == '__main__':
    app.run(debug=True) 
    
    
## 체크섬 폴더 만들어서 걔 QUEUE에 큐안에서 다 되는대로 뿌려주기.