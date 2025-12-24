import os
from flask import Flask, request, jsonify
from openai import OpenAI, api_client
from dotenv import load_dotenv
from flask_cors import CORS
import json

# تحميل متغيرات البيئة (مهم لبيئة التطوير المحلية)
load_dotenv()

app = Flask(__name__)
# تمكين CORS للسماح بالطلبات من أي مصدر
CORS(app)

# --- بداية الكود المعدل ---
# إعداد عميل OpenAI مع معالجة خطأ 'proxies'
try:
    # الطريقة القياسية
    client = OpenAI()
except TypeError as e:
    # هذا الجزء سيعمل إذا واجهنا خطأ الوسيط 'proxies' في بيئة مثل Render
    if 'proxies' in str(e):
        print("Ignoring 'proxies' argument for OpenAI client initialization.")
        # نقوم بتهيئة العميل يدوياً بدون تمرير الوسائط غير المتوقعة
        # هذا يتطلب استيراد http_client من openai
        from openai._base_client import DefaultHttpxClient
        
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY" ),
            http_client=DefaultHttpxClient(proxies={} ) # تمرير بروكسي فارغ
        )
    else:
        # إذا كان الخطأ من نوع آخر، أظهره
        raise e
except Exception as e:
    # هذا سيساعد في تشخيص المشكلة إذا لم يتم تعيين مفتاح API
    print(f"Error initializing OpenAI client: {e}")
    client = None
# --- نهاية الكود المعدل ---


@app.route('/')
def home():
    """A simple route to check if the backend is running."""
    return "Backend for PDF Quiz Generator is running!"


@app.route('/generate-questions', methods=['POST'])
def generate_questions_api():
    """
    API endpoint to generate quiz questions from text using an LLM.
    """
    if not client:
        return jsonify({"error": "OpenAI client not initialized. Check API key or server logs."}), 500
        
    try:
        data = request.json
        full_text = data.get('text')

        if not full_text or len(full_text.strip()) < 20:
            return jsonify({"error": "النص المقدم قصير جدًا أو فارغ."}), 400

        system_prompt = """
        You are an expert AI assistant. Your task is to analyze the provided text and extract Multiple Choice Questions (MCQs).
        Follow these rules strictly:
        1. Only extract questions that are clearly present in the text. Do not invent questions or information.
        2. For each question, you must identify the question text, four options (A, B, C, D), and the correct answer.
        3. If a question is a True/False question, make the options ["True", "False"].
        4. The correct answer must be the exact text of one of the options.
        5. Return the result as a JSON object containing a single key "questions" which holds a list of question objects.
        6. The format for each question object must be:
           {
             "question": "The question text...",
             "options": ["Option A", "Option B", "Option C", "Option D"],
             "answer": "The correct answer (must match one of the options)"
           }
        7. If you find no questions in the text, return an empty list: {"questions": []}.
        8. Do not add any extra text, explanations, or markdown formatting before or after the JSON object.
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        response_content = response.choices[0].message.content
        
        try:
            response_data = json.loads(response_content)
            
            if "questions" not in response_data or not isinstance(response_data["questions"], list):
                raise ValueError("The 'questions' key is missing or is not a list in the AI response.")

            return jsonify(response_data["questions"])

        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            print(f"Received response from API: {response_content}")
            return jsonify({"error": "Failed to parse the response from the AI.", "details": str(e)}), 500

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
