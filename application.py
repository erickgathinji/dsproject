from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import traceback

application = Flask(__name__)

app = application


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score")),
            )
            
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('index.html', results=results[0])
        
      
        except Exception as e:
            # Safely capture the full traceback error text
            error_trace = traceback.format_exc()
            return f"<h3>Application Error:</h3><p>{str(e)}</p><pre>{error_trace}</pre>", 500
        
    
        
if __name__=="__main__":
    app.run(host="0.0.0.0")  
    
    
    
