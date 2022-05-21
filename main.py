import pandas as pd
from flask import Flask, render_template, request, make_response, session
import numpy as np
import cv2
import base64
from algorithms.orb_algorithm import orb_
from algorithms.modified_lambda import modified_lambda_
#from algorithms.modified_orb import base
from algorithms.extract_orb_feats import orb_new
import json
import pdb

app = Flask(__name__)
app.secret_key = "wydhajbndi28hhodik"


@app.route("/")
def home():
    return render_template("main.html")

@app.route("/algorithm")
def orb():
    #pdb.set_trace()
    alg = request.args.get("algorithm")
    if alg == "fast":
        name = "FAST - Features from Accelerated and Segments Test"
    elif alg=="mfast":
        name = "MFAST - Modified Features from Accelerated and Segments Test"
    elif alg == "mlambda":
        name = "Modified Lambda detection"
    elif alg == "orb":
        name = "ORB 2.0 - Oriented FAST and Rotated BRIEF"
    return render_template("orb.html", algorithm=name)

@app.route("/upload", methods=["POST"])
def upload():
    #pdb.set_trace()
    alg = request.args.get("algorithm")
    download = request.args.get("download")
    file = request.files["file"]
    binary_image = file.read()
    binary_image_display = base64.b64encode(binary_image).decode("utf-8")
    if not file:
        return render_template("main.html")
    np_array = np.frombuffer(binary_image, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    if "MFAST" in alg:
        name = "MFAST - Modified Features from Accelerated and Segments Test"
        #processed_image_ = base(image)
    elif "FAST" in alg:
        name = "FAST - Features from Accelerated and Segments Test"
        processed_image_ = orb_(image)
    elif "Lambda" in alg:
        name = "Modified Lambda detection"
        processed_image_ = modified_lambda_(image)
    elif "ORB" in alg:
        name = "ORB 2.0 - Oriented FAST and Rotated BRIEF"
        processed_image_ = orb_new(image)
    processed_image = processed_image_[0]
    number_kp = str(processed_image_[1])+" Key points"
    df = processed_image_[2]
    if download:
        resp = make_response(df.to_csv(index=True))
        resp.headers["Content-Disposition"] = "attachment; filename=Peaks.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    _, processed_image = cv2.imencode(".jpeg", processed_image)
    processed_image = base64.b64encode(processed_image).decode("utf-8")

    return render_template("orb.html",img_og=binary_image_display, img_processed=processed_image,
                           number_kp=number_kp, algorithm=name)




@app.route("/download")
def download():
    df=[]
    df = session["df"]
    df = json.loads(df)
    df = pd.DataFrame(df["data"])
    resp = make_response(df.to_csv(index=False))
    resp.headers["Content-Disposition"]="attachment; filename=Peaks.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


if __name__ == "__main__":
    app.run(debug=True)



#buff = BytesIO()
#pil_image = Image.fromarray(image_og_show)
#pil_image.save(buff, format="JPEG")
#return_image = base64.b64encode(buff.getvalue()).decode("utf-8")


