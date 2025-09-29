from flask import Flask, render_template, request, jsonify
from groq import Groq
import os
from datetime import datetime

app = Flask(__name__)

# ============ GROQ API KEY CONFIGURATION ============
# Initialize Groq client with API key from environment variable
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Available Groq models (choose the best one for your needs)
MODEL = "llama-3.3-70b-versatile"  # Fast and capable model

# Verify API key is loaded
if not os.getenv('GROQ_API_KEY'):
    print("Warning: GROQ_API_KEY environment variable not found!")
    print("Make sure you've set it with: set GROQ_API_KEY=your-key-here")
    print("Get your free API key from: https://console.groq.com/keys")
# =============================================

# Updated Resume data for context
RESUME_CONTEXT = """
VELLANKI BHARATH's Resume:

CONTACT INFORMATION:
- Location: Substation street KOTHAGANDIGUDEM, ELURU district, Andhra Pradesh
- Phone: 7989384355
- Email: vellankibharath.b@gmail.com
- LinkedIn: https://www.linkedin.com/in/vellankibharath

SUMMARY:
Resolute and innovative Computer Science Engineering graduate. Enthusiastic about leveraging AI technologies to solve complex problems and drive technological advancements. Continuously expanding knowledge in AI tools to work fast and efficient. Solid foundation in Computer Science fundamentals, Effective communicator, and collaborator across teams.

EDUCATION:
- Amrita Sai Institute of Science and Technology, Vijayawada (Dec 2021 - May 2025) - GRADUATED
  Bachelor of Technology in Computer Science and Engineering - 7.8 CGPA - COMPLETED
- SASI junior college, KAMAVARAPUKOTA (June 2019 - April 2021)
  Board of Intermediate - 88.8%
- SASI English Medium School, KAMAVARAPUKOTA (June 2018 - April 2019)
  Secondary School Certificate - 9.7 GPA

PROGRAMMING SKILLS:
- Python
- Pandas, Scikit-learn (Libraries)
- Data visualization (Matplotlib, Seaborn)
- Machine Learning (Algorithms)
- SQL
- HTML, CSS, JavaScript
- Flask

NON-PROGRAMMING SKILLS:
- AI tools

TOOLS:
VS Code, Jupyter Notebook, GitHub, Postman, Docker, Docker Hub

PROFESSIONAL EXPERIENCE:
ADEPT TALENT ACQUISITION - Data Annotator | Internship (April 2025 - Present)
- Developed ML Labelling for predictive analytics
- Collaborated with cross-functional teams to develop data-driven solutions

PROJECTS:
1. HEART STROKE PREDICTION
   - Used Kaggle dataset (42,000 samples)
   - Performed Data cleaning, Feature Engineering, Dimensionality Reduction, Outlier removal
   - Tested 6 regression algorithms for accuracy
   - Deployed using Flask, HTML, CSS, JavaScript
   - GitHub: https://github.com/VellankiBharath/Heart_Stroke_Prediction

2. TEXT TO IMAGE GENERATION
   - Built using Stable Diffusion for image generation
   - Used CLIP for image analysis, SAM2 for segmentation
   - Flask server tested with Postman
   - Frontend: HTML, CSS, JavaScript
   - Dockerized application
   - GitHub: https://github.com/VellankiBharath/Deep_Edge/tree/main/Deep_Edge/deep_edge

CERTIFICATIONS:
- Python for Data Science - Swayam
- Data Structure using Python - Great Learning
- SQL for Data Science - Great Learning
- Version Control Git and GitHub - Great Learning
- Pandas, NumPy, Data Visualisation - Great Learning
- Machine Learning - Great Learning
- AI tools - B 10x

ACHIEVEMENTS:
- Certificate of appreciation by Government of Andhra Pradesh (Acts as mentor for government teachers, managing IFPs)
- Certificate of Participation in Project Expo by Amrita Sai Institute of Science & Technology

GRADUATION STATUS: Vellanki Bharath has COMPLETED his Bachelor of Technology in Computer Science and Engineering from Amrita Sai Institute of Science and Technology in May 2025. He is a GRADUATE, not a current student.
"""

def get_chatbot_response(user_question):
    """Generate AI response about the resume using Groq API"""
    try:
        system_prompt = f"""
        You are a helpful chatbot that answers questions about Vellanki Bharath's resume. 
        Be friendly, professional, and informative. Answer questions directly based on the resume information provided.
        
        IMPORTANT: Vellanki Bharath has ALREADY GRADUATED in May 2025. He completed his B.Tech in Computer Science and Engineering. 
        He is NOT a current student - he is a GRADUATE. Always refer to him as a graduate, not as a student.
        
        When asked about graduation:
        - He GRADUATED in May 2025
        - He COMPLETED his Bachelor of Technology in Computer Science and Engineering 
        - He is a COMPUTER SCIENCE GRADUATE
        - He has a 7.8 CGPA
        
        If asked about something not in the resume, politely mention that information isn't available in the resume, 
        but direct them to contact Bharath directly at vellankibharath.b@gmail.com for additional information.
        
        Resume Information:
        {RESUME_CONTEXT}
        """
        
        # Groq API call - different syntax than OpenAI
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=1,
            stream=False,  # Set to True for streaming responses
            stop=None
        )
        
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        error_message = str(e)
        if "rate_limit" in error_message.lower() or "quota" in error_message.lower():
            return """I'm currently experiencing high demand. Please try again in a moment or 
                     contact Bharath directly at vellankibharath.b@gmail.com for information 
                     about his resume and qualifications."""
        else:
            return f"Sorry, I'm having trouble processing your request. Please contact Bharath at vellankibharath.b@gmail.com for assistance."

def get_fallback_response(user_question):
    """Fallback responses when API is unavailable"""
    question_lower = user_question.lower()
    
    if any(word in question_lower for word in ['contact', 'email', 'phone', 'reach']):
        return "📧 Email: vellankibharath.b@gmail.com\n📱 Phone: 7989384355\n🔗 LinkedIn: https://www.linkedin.com/in/vellankibharath"
    
    elif any(word in question_lower for word in ['experience', 'work', 'job', 'internship']):
        return "🏢 Currently working as Data Annotator at Adept Talent Acquisition (April 2025 - Present)\n• Developed ML Labelling for predictive analytics\n• Collaborated with cross-functional teams"
    
    elif any(word in question_lower for word in ['education', 'college', 'degree', 'study', 'graduate', 'graduation']):
        return "🎓 GRADUATED with B.Tech in Computer Science - Amrita Sai Institute of Science and Technology\n📊 CGPA: 7.8 (Completed in May 2025)\n✅ Computer Science Graduate\n📚 Intermediate: 88.8% from SASI Junior College"
    
    elif any(word in question_lower for word in ['skills', 'programming', 'python', 'technology']):
        return "💻 Programming: Python, SQL, HTML/CSS/JavaScript, Flask\n🤖 AI/ML: Pandas, Scikit-learn, Machine Learning, Data Visualization\n🛠️ Tools: VS Code, Jupyter, GitHub, Docker, Postman"
    
    elif any(word in question_lower for word in ['projects', 'github', 'portfolio']):
        return "🚀 Key Projects:\n1️⃣ Heart Stroke Prediction (42K dataset, 6 ML algorithms, Flask deployment)\n2️⃣ Text to Image Generation (Stable Diffusion, CLIP, SAM2, Dockerized)\n🔗 GitHub: Available in resume"
    
    elif 'year' in question_lower and any(word in question_lower for word in ['graduate', 'graduation', 'finish', 'complete']):
        return "🎓 Vellanki Bharath GRADUATED in May 2025 with a B.Tech in Computer Science and Engineering from Amrita Sai Institute of Science and Technology. He completed his degree with a 7.8 CGPA."
    
    else:
        return "Hi! I'm Bharath's resume assistant. Bharath is a Computer Science Graduate (May 2025). Ask me about his:\n• 📞 Contact Information\n• 💼 Work Experience\n• 🎓 Education & Graduation\n• 💻 Technical Skills\n• 🚀 Projects\n\nFor detailed discussions, reach out to vellankibharath.b@gmail.com"

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Try Groq API first, fallback to rule-based responses
    try:
        bot_response = get_chatbot_response(user_message)
    except:
        bot_response = get_fallback_response(user_message)
    
    return jsonify({
        'response': bot_response,
        'timestamp': datetime.now().strftime('%H:%M'),
        'model': MODEL
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Resume chatbot is running with Groq!',
        'model': MODEL,
        'api': 'Groq'
    })

# Optional: Add model switching capability
@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch between available Groq models"""
    global MODEL
    
    available_models = {
        'llama-fast': 'llama-3.1-8b-instant',      # Fastest responses
        'llama-balanced': 'llama-3.3-70b-versatile', # Best balance
        'llama-powerful': 'llama-3.1-405b-reasoning', # Most capable
        'mixtral': 'mixtral-8x7b-32768'            # Alternative model
    }
    
    requested_model = request.json.get('model_key', 'llama-balanced')
    
    if requested_model in available_models:
        MODEL = available_models[requested_model]
        return jsonify({'success': True, 'new_model': MODEL})
    else:
        return jsonify({'success': False, 'error': 'Invalid model'}), 400

if __name__ == '__main__':
    print(f"🚀 Starting Resume Chatbot with Groq API")
    print(f"🤖 Model: {MODEL}")
    print(f"🌐 Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
